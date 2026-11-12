"""SCF electron-role ledger, separate from correlated quality observations.

Only an authenticated integer-occupation SCF or audited one-configuration ROHF
creates roles. Later bases carry those labels by maximum AO-overlap assignment
inside the actual calculation's partitions. The complete mapping chain is
replayable from immutable files.
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import linear_sum_assignment

from .common import file_hash, read_json
from .orbital_io import read_inporb, read_openmolcas_orbital_hdf5, read_rasorb_coefficients

GROUPS = ("frozen", "inactive", "active", "secondary", "deleted")
SCHEMA = "scf_integer_roles_partition_overlap_v1"
SINGLE_CONFIGURATION_ROHF = "native_single_configuration_rohf_v1"


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=file_hash(path))


def verified(record):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("Electron-role source requires an explicit file hash binding")
    path = Path(record["path"])
    if file_hash(path) != record["sha256"]:
        raise ValueError("Electron-role source artifact changed: " + str(path))
    return path


def partition(context):
    used = set(context.active + context.inactive + context.frozen + context.deleted)
    return dict(
        active=list(context.active),
        inactive=list(context.inactive),
        frozen=list(context.frozen),
        deleted=list(context.deleted),
        secondary=sorted(set(range(context.nmo)) - used),
        electrons=context.electrons,
    )


def validate_partition(roles, parts):
    if any(type(r) is not int or r not in (0, 1, 2) for r in roles) or sorted(
        i for group in GROUPS for i in parts[group]
    ) != list(range(len(roles))):
        raise ValueError("Invalid electron-role partition")
    if sum(roles[i] for i in parts["active"]) != parts["electrons"]:
        raise ValueError("Candidate SCF electron roles do not sum to the active electron count")
    if any(roles[i] != 2 for group in ("inactive", "frozen") for i in parts[group]) or any(
        roles[i] != 0 for group in ("secondary", "deleted") for i in parts[group]
    ):
        raise ValueError("Inactive/secondary electron roles must be 2/0; role 1 must remain active")


def _basis(record, kind):
    path = verified(record)
    if kind in ("native", SINGLE_CONFIGURATION_ROHF):
        mo = read_openmolcas_orbital_hdf5(path)
        with h5py.File(path) as h:
            coordinates = np.asarray(h["CENTER_COORDINATES"][()], float)
        data = dict(
            c=mo.mo_coefficients,
            s=mo.ao_overlap,
            ids=mo.basis_function_ids,
            atoms=mo.center_labels,
            geometry=coordinates,
            parts={
                name: [i for i, t in enumerate(mo.type_indices) if t == tag]
                for name, tag in zip(GROUPS, ("f", "i", "2", "s", "d"))
            },
        )
    elif kind == "pyscf":
        with np.load(path, allow_pickle=False) as z:
            data = dict(
                c=z["coefficients"],
                s=z["overlap"],
                ids=np.arange(z["coefficients"].shape[0]),
                atoms=(),
                geometry=np.empty((0, 3)),
            )
    else:
        raise ValueError("Unsupported electron-role basis format")
    c, s = data["c"], data["s"]
    if not np.isfinite(c).all() or not np.allclose(
        c.T @ s @ c, np.eye(c.shape[1]), atol=1e-7, rtol=0
    ):
        raise ValueError("Electron-role basis is not AO-metric orthonormal")
    return data


def _single_configuration_rohf(reference, h):
    """A one-electron, one-orbital RASSCF is an explicitly audited ROHF source."""
    from .backend import _converged

    paths = {
        key: verified(reference[key])
        for key in ("checkpoint", "solver_input", "solver_log", "execution")
    }
    execution = read_json(paths["execution"])
    if not _converged(
        paths["solver_log"].read_text(errors="replace"), "optimize", execution["return_code"]
    ):
        raise ValueError("Single-configuration ROHF optimization did not converge")
    text = "\n".join(
        line
        for line in paths["solver_input"].read_text().splitlines()
        if not line.lstrip().startswith("*")
    )
    blocks = re.findall(r"&RASSCF\b([^&]*)", text, re.I)
    if len(blocks) != 1:
        raise ValueError("Single-configuration ROHF requires exactly one RASSCF input")
    block = blocks[0]
    required = {"spin": [2], "nactel": [1, 0, 0], "ras2": [1]}
    for key, expected in required.items():
        values = re.findall(r"^\s*" + key + r"\s*=\s*([^\n]+)", block, re.I | re.M)
        if len(values) != 1 or values[0].split() != [str(v) for v in expected]:
            raise ValueError("Single-configuration ROHF input differs: " + key)
    roots = re.findall(r"^\s*ciroot\s*=\s*([^\n]+)", block, re.I | re.M)
    if len(roots) != 1 or re.split(r"[\s,;]+", roots[0].strip()) not in (
        ["1", "1", "1"],
        ["1", "1", "1", "1"],
    ):
        raise ValueError("Single-configuration ROHF input must request one root")
    if re.search(r"^\s*(?:cionly|dmrg)\b", block, re.I | re.M):
        raise ValueError("Single-configuration ROHF must optimize its orbitals")
    for key in ("NACTEL", "NROOTS", "NSTATES"):
        if int(h.attrs[key]) != 1:
            raise ValueError("Single-configuration ROHF requires one electron and one state")
    if int(h.attrs["SPINMULT"]) != 2:
        raise ValueError("Single-configuration ROHF requires a doublet")
    if np.asarray(h.attrs["STATE_ROOTID"]).reshape(-1).tolist() != [1] or not np.array_equal(
        np.asarray(h.attrs["STATE_WEIGHT"]).reshape(-1), [1.0]
    ):
        raise ValueError("Single-configuration ROHF state identity differs")
    ci = np.asarray(h["CI_VECTORS"][()], float)
    if (
        ci.size != 1
        or not np.isfinite(ci).all()
        or not np.isclose(np.sum(ci**2), 1.0, atol=1e-8, rtol=0)
    ):
        raise ValueError("Single-configuration ROHF requires one normalized CI coefficient")


def _integer_scf(reference):
    kind = reference.get("format", "native")
    data = _basis(reference["basis_file"], kind)
    if kind in ("native", SINGLE_CONFIGURATION_ROHF):
        path = verified(reference["basis_file"])
        with h5py.File(path) as h:
            module = h.attrs.get("MOLCAS_MODULE", "")
            if isinstance(module, bytes):
                module = module.decode()
            if kind == SINGLE_CONFIGURATION_ROHF:
                if str(module).strip().upper() != "RASSCF":
                    raise ValueError(
                        "Single-configuration ROHF must retain its RASSCF module identity"
                    )
                _single_configuration_rohf(reference, h)
            elif str(module).strip().upper() != "SCF":
                raise ValueError(
                    "Electron roles must originate from an SCF HDF5, not correlated occupations"
                )
        mo = read_openmolcas_orbital_hdf5(path)
        tags = mo.type_indices
        if "checkpoint" in reference:
            checkpoint = verified(reference["checkpoint"])
            inp = read_inporb(checkpoint)
            coeff, checkpoint_tags = read_rasorb_coefficients(
                checkpoint, symmetry_block_sizes=mo.symmetry_block_sizes
            )
            if (
                checkpoint_tags != tags
                or inp.type_indices != tags
                or not np.allclose(
                    mo.native_coefficients_to_ao(coeff), data["c"], atol=1e-8, rtol=0
                )
                or not np.allclose(inp.occupations, mo.occupations, atol=1e-6, rtol=0)
            ):
                raise ValueError("SCF electron-role HDF5/checkpoint differ")
        occupations = mo.occupations
        if kind == SINGLE_CONFIGURATION_ROHF and (
            tags.count("2") != 1
            or any(tag not in ("i", "2", "s") for tag in tags)
            or sum(abs(value - 1.0) <= 1e-6 for value in occupations) != 1
            or any(
                abs(value - {"i": 2.0, "2": 1.0, "s": 0.0}[tag]) > 1e-6
                for value, tag in zip(occupations, tags)
            )
        ):
            raise ValueError(
                "Single-configuration ROHF needs one singly occupied active orbital and integer occupations"
            )
        frozen = [i for i, tag in enumerate(tags) if tag == "f"]
        deleted = [i for i, tag in enumerate(tags) if tag == "d"]
    else:
        with np.load(verified(reference["basis_file"]), allow_pickle=False) as z:
            if str(z["source_kind"]) != "pyscf_converged_scf" or not bool(z["converged"]):
                raise ValueError("Electron roles require a converged PySCF SCF artifact")
            occupations = z["occupations"]
        frozen, deleted = [], []
    # Integer classification is performed only on the verified SCF output.
    roles = []
    for value in occupations:
        matches = [r for r in (0, 1, 2) if abs(value - r) <= 1e-6]
        if len(matches) != 1:
            raise ValueError("SCF electron roles require explicit integer 0/1/2 occupations")
        roles.append(matches[0])
    active = list(reference["active"])
    if any(type(i) is not int for i in active) or len(set(active)) != len(active):
        raise ValueError("SCF role-reference active indices must be distinct integers")
    if not set(active) <= set(range(len(roles))) - set(frozen + deleted):
        raise ValueError("SCF role-reference active indices are outside the eligible basis")
    parts = dict(
        active=sorted(active),
        frozen=frozen,
        deleted=deleted,
        inactive=[i for i, r in enumerate(roles) if r == 2 and i not in active + frozen],
        secondary=[i for i, r in enumerate(roles) if r == 0 and i not in active + deleted],
        electrons=sum(roles[i] for i in active),
    )
    validate_partition(roles, parts)
    return roles, parts, data


def _map(roles, old, new, before, after, *, geometry_overlap=None):
    validate_partition(roles, before)
    if "parts" in new and any(sorted(after[group]) != new["parts"][group] for group in GROUPS):
        raise ValueError("Electron-role output partition differs from the actual orbital file")
    if (
        old["c"].shape != new["c"].shape
        or old["atoms"] != new["atoms"]
        or not np.array_equal(old["ids"], new["ids"])
        or (
            geometry_overlap is None
            and (
                not np.allclose(old["geometry"], new["geometry"], atol=1e-8, rtol=0)
                or not np.allclose(old["s"], new["s"], atol=1e-8, rtol=0)
            )
        )
    ):
        raise ValueError("Electron-role transport changed the AO basis or geometry")
    cross = (
        old["c"].T @ old["s"] @ new["c"]
        if geometry_overlap is None
        else np.asarray(geometry_overlap, float)
    )
    if (
        cross.shape != (len(roles), len(roles))
        or not np.isfinite(cross).all()
        or np.max(abs(cross)) > 1 + 1e-6
    ):
        raise ValueError("Invalid verified electron-role overlap")
    mapping = [-1] * len(roles)
    strengths = [0.0] * len(roles)
    output = [-1] * len(roles)
    for group in GROUPS:
        a, b = sorted(before[group]), sorted(after[group])
        if len(a) != len(b):
            raise ValueError("Electron-role transport changed partition dimensions: " + group)
        rows, cols = linear_sum_assignment(-(np.abs(cross[np.ix_(a, b)]) ** 2))
        for row, col in zip(rows, cols):
            i, j = a[int(row)], b[int(col)]
            mapping[i] = j
            strengths[i] = float(abs(cross[i, j]) ** 2)
            output[j] = roles[i]
    validate_partition(output, after)
    return output, dict(old_to_new=mapping, matched_squared_overlaps=strengths)


def _step(proof, roles, old, input_basis, before, output_basis, after):
    new = _basis(output_basis, proof["reference"].get("format", "native"))
    roles, mapping = _map(roles, old, new, before, after)
    step = dict(
        input_basis=input_basis,
        output_basis=output_basis,
        input_partition=copy.deepcopy(before),
        output_partition=copy.deepcopy(after),
        **mapping,
    )
    proof["steps"].append(step)
    return roles, proof


def from_reference(reference, output_basis, output_partition):
    reference = copy.deepcopy(reference)
    roles, before, old = _integer_scf(reference)
    proof = dict(schema=SCHEMA, reference=reference, steps=[])
    return _step(proof, roles, old, reference["basis_file"], before, output_basis, output_partition)


def replay(proof, output_basis, output_partition, expected_roles=None):
    if proof.get("schema") != SCHEMA or not proof.get("steps"):
        raise ValueError("Missing verified SCF electron-role provenance")
    roles, reference_partition, old = _integer_scf(proof["reference"])
    if proof["steps"][0]["input_partition"] != reference_partition:
        raise ValueError(
            "First electron-role mapping does not use the original SCF initial partition"
        )
    previous = proof["reference"]["basis_file"]
    for step in proof["steps"]:
        if step["input_basis"] != previous:
            raise ValueError("Electron-role mapping chain is discontinuous")
        new = _basis(step["output_basis"], proof["reference"].get("format", "native"))
        overlap = None
        if "geometry_overlap" in step:
            from .geometry_transfer import bound_overlap

            overlap = bound_overlap(
                step["input_basis"], step["output_basis"], step["geometry_overlap"]
            )
        roles, mapping = _map(
            roles,
            old,
            new,
            step["input_partition"],
            step["output_partition"],
            geometry_overlap=overlap,
        )
        if step["old_to_new"] != mapping["old_to_new"] or not np.allclose(
            step["matched_squared_overlaps"],
            mapping["matched_squared_overlaps"],
            atol=1e-12,
            rtol=0,
        ):
            raise ValueError("Electron-role overlap mapping changed")
        previous, old = step["output_basis"], new
    if previous != output_basis or proof["steps"][-1]["output_partition"] != output_partition:
        raise ValueError("Electron-role provenance is bound to a different output basis/partition")
    if expected_roles is not None and list(expected_roles) != roles:
        raise ValueError("Context electron roles disagree with the verified SCF transport")
    return roles


def from_context(context, input_partition, output_basis, output_partition):
    proof = copy.deepcopy(context.payload.get("electron_role_provenance", {}))
    old_binding = binding(context.payload["basis_file"])
    roles = replay(proof, old_binding, partition(context), context.electron_roles)
    old = _basis(old_binding, proof["reference"].get("format", "native"))
    return _step(proof, roles, old, old_binding, input_partition, output_basis, output_partition)


def from_geometry_context(
    context, input_partition, output_basis, output_partition, parent_molden, child_molden
):
    """Append a hash-bound cross-geometry step without weakening ordinary maps."""
    from .geometry_transfer import bound_overlap

    proof = copy.deepcopy(context.payload.get("electron_role_provenance", {}))
    if proof.get("reference", {}).get("format", "native") != "native":
        raise ValueError("Geometry role transfer currently requires native integer SCF provenance")
    old_binding = binding(context.payload["basis_file"])
    roles = replay(proof, old_binding, partition(context), context.electron_roles)
    evidence = dict(
        schema="pas_cross_geometry_ao_overlap_v1",
        parent_molden=parent_molden,
        child_molden=child_molden,
    )
    cross = bound_overlap(old_binding, output_basis, evidence)
    roles, mapping = _map(
        roles,
        _basis(old_binding, "native"),
        _basis(output_basis, "native"),
        input_partition,
        output_partition,
        geometry_overlap=cross,
    )
    proof["steps"].append(
        dict(
            input_basis=old_binding,
            output_basis=output_basis,
            input_partition=copy.deepcopy(input_partition),
            output_partition=copy.deepcopy(output_partition),
            geometry_overlap=evidence,
            **mapping,
        )
    )
    return roles, proof


def verify_context(context):
    return replay(
        context.payload.get("electron_role_provenance", {}),
        binding(context.payload["basis_file"]),
        partition(context),
        context.electron_roles,
    )


def discover_reference(seed, explicit=None):
    """Only explicit, hash-bound cold-stage source metadata is searched."""
    if explicit is not None:
        return copy.deepcopy(explicit)
    initial = seed.get("initialization", {})
    if "electron_role_reference" in initial:
        return copy.deepcopy(initial["electron_role_reference"])
    if "electron_role_reference" in seed:
        return copy.deepcopy(seed["electron_role_reference"])
    shared = initial.get("shared_initial_guess")
    if shared is not None:
        raw = read_json(verified(shared))
        sources = raw["source_files"]
        return dict(
            basis_file=sources["foundation_hdf5"],
            checkpoint=sources["foundation_checkpoint"],
            active=raw["active"],
        )
    raise ValueError(
        "Missing SCF electron-role source: provide role_reference with bound SCF files and original initial active indices"
    )
