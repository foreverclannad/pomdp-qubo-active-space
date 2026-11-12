"""Standalone, receipt-bound OpenMolcas and Block2 physical actions.

No legacy controller, scores, orbital-role constraints or belief is imported.
All physical actions are restricted to node-local cu07 scratch.
"""

from __future__ import annotations

import copy
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from contextvars import ContextVar
from fractions import Fraction
from functools import reduce
from math import gcd, lcm
from pathlib import Path

import h5py
import numpy as np

from . import metrics
from .common import file_hash, identity, read_json, write_json
from .composition import determinant_pair_matrix, harmonic_channel_data
from .config import configuration
from .contracts import Context
from .orbital_io import (
    _replace_type_indices,
    ao_subspace_projection_scores,
    read_inporb,
    read_openmolcas_orbital_hdf5,
    read_rasorb_coefficients,
    select_ao_indices,
)

MOLCAS = "/nfs-soft/OpenMolcas-block2tmp/v4"
CAS_MOLCAS = "/nfs-soft/OpenMolcas-git/v4"
BLOCK2MAIN = "/home/lichen/.venvs/pomdp-aos/bin/block2main"
_RECONCILE_ONLY = ContextVar("pas_reconcile_only", default=False)


def _allow_dispatch():
    if _RECONCILE_ONLY.get():
        raise RuntimeError(
            "Unresolved physical stage: recovery cannot launch a solver or evidence job"
        )


class ContractError(ValueError):
    """An artifact, state or method contract differs; never a quality label."""


def binding(path):
    return dict(path=str(Path(path).resolve()), sha256=file_hash(path))


def verify_binding(record):
    if file_hash(record["path"]) != record["sha256"]:
        raise ContractError("Artifact changed: " + record["path"])
    return Path(record["path"])


def runtime():
    from importlib.metadata import version

    import block2

    if version("block2") != "0.5.3":
        raise RuntimeError("This adapter requires audited Block2 0.5.3")
    return dict(
        block2_version="0.5.3",
        python=sys.executable,
        block2_library=binding(block2.__file__),
        block2main=binding(BLOCK2MAIN),
        pymolcas=binding(Path(MOLCAS) / "pymolcas"),
    )


def checked_directory(directory, threads):
    path = Path(directory).resolve()
    configured = os.environ.get("PAS_WORK_ROOT")
    root = Path("/tmp" if configured is None else configured)
    if not root.is_absolute() or root == Path(root.anchor) or not root.is_dir():
        raise RuntimeError("Native work root must be an existing absolute directory")
    root = root.resolve()
    if root == Path(root.anchor):
        raise RuntimeError("Native work root cannot be a filesystem root")
    if configured is not None and root != Path("/tmp") and root.stat().st_uid != os.getuid():
        raise RuntimeError("Configured native work root must be owned by the current user")
    if socket.gethostname().split(".")[0] != "cu07" or not path.is_relative_to(root):
        raise RuntimeError(
            "Native physical execution requires cu07 and the configured work root: " + str(root)
        )
    if not 1 <= threads <= 32:
        raise ContractError("Per-task CPU limit is 32")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _states(states):
    states = copy.deepcopy(states)
    roots = states.get("roots", [])
    if not roots or any(type(root) is not int or root < 1 for root in roots):
        raise ContractError("Native adapter requires positive integer state roots")
    states.setdefault("ids", ["root_" + str(r) for r in states["roots"]])
    if states["roots"] != list(range(1, len(states["roots"]) + 1)):
        raise ContractError("Native adapter requires consecutive lowest requested roots")
    metrics._weights(states["weights"], len(states["roots"]))
    if len(states["ids"]) != len(states["roots"]) or len(set(states["ids"])) != len(states["ids"]):
        raise ContractError("Invalid state identities")
    return states


def read_densities(path, states, electrons, *, require_natural=True, require_ci=True):
    """Full per-root density in the explicitly verified Molcas MO gauge."""
    states = _states(states)
    mo = read_openmolcas_orbital_hdf5(path)
    if len(mo.symmetry_block_sizes) != 1 or set(mo.type_indices) - {"f", "i", "2", "s", "d"}:
        raise ContractError("Density adapter supports NOSYM full CAS only")
    active = [i for i, t in enumerate(mo.type_indices) if t == "2"]
    k = len(active)
    with h5py.File(path) as h:
        if "DENSITY_MATRIX" not in h or (require_ci and "CI_VECTORS" not in h):
            raise ContractError("Complete native CAS root densities/CI provenance are unavailable")
        attrs = h.attrs
        if (
            int(attrs["NACTEL"]) != electrons
            or int(attrs["SPINMULT"]) != states["spin_multiplicity"]
            or np.asarray(attrs["STATE_ROOTID"]).reshape(-1).tolist() != states["roots"]
            or not np.allclose(
                np.asarray(attrs["STATE_WEIGHT"]).reshape(-1), states["weights"], atol=1e-12, rtol=0
            )
        ):
            raise ContractError("HDF5 state/electron contract differs")
        raw = np.asarray(h["DENSITY_MATRIX"][()], float)
        if raw.shape not in ((len(states["roots"]), k * k), (len(states["roots"]), k, k)):
            raise ContractError(
                "Unsupported density layout; cannot guess root/orbital correspondence: "
                + str(raw.shape)
            )
        d = raw.reshape(len(states["roots"]), k, k)
        energies = np.asarray(h["ROOT_ENERGIES"][()], float).reshape(-1)
    metrics.validate_densities(d, states["weights"], electrons)
    if require_natural and not np.allclose(
        np.einsum("r,rij->ij", states["weights"], d),
        np.diag(mo.occupations[active]),
        atol=1e-6,
        rtol=0,
    ):
        raise ContractError("Density is not bound to the HDF5 state-average natural MO basis")
    if energies.shape != (len(states["roots"]),) or not np.all(np.isfinite(energies)):
        raise ContractError("Missing complete finite root energies")
    return d, energies


def _geometry(case, mo, path):
    lines = case["geometry_xyz"].splitlines()
    atoms = [line.split() for line in lines[2:] if line.strip()]
    if len(atoms) != int(lines[0]):
        raise ContractError("Malformed case geometry")
    symbols = [a[0] for a in atoms]
    xyz = np.array([a[1:4] for a in atoms], float)
    labels = [re.match(r"[A-Za-z]+", s).group(0) for s in mo.center_labels]
    if [s.lower() for s in symbols] != [s.lower() for s in labels]:
        raise ContractError("MO atoms differ from case geometry")
    with h5py.File(path) as h:
        coordinates = np.asarray(h["CENTER_COORDINATES"][()], float)
    if coordinates.shape != xyz.shape or not np.allclose(
        coordinates * 0.529177210903, xyz, atol=1e-7, rtol=0
    ):
        raise ContractError("HDF5 coordinates differ from case geometry")


def bound_geometry_text(context):
    """Validated input AO centres, independent of a build's Angstrom constant."""
    source = Path(context.payload["basis_file"])
    case = context.payload["case"]
    if (
        file_hash(source) != context.basis_id
        or identity(case["geometry_xyz"]) != context.geometry_id
    ):
        raise ContractError("Geometry source differs from its bound orbital context")
    mo = read_openmolcas_orbital_hdf5(source)
    _geometry(case, mo, source)
    with h5py.File(source) as handle:
        coordinates = np.asarray(handle["CENTER_COORDINATES"][()], float)
    symbols = [re.match(r"[A-Za-z]+", label).group(0) for label in mo.center_labels]
    # XYZ comment "bohr" tells GATEWAY to use atomic units. Seventeen digits
    # round-trip the HDF5 float64 values instead of converting the old XYZ again.
    return (
        str(len(symbols))
        + "\nbohr\n"
        + "".join(
            symbol + " " + " ".join(format(v, ".17g") for v in row) + "\n"
            for symbol, row in zip(symbols, coordinates, strict=True)
        )
    )


def write_bound_geometry(context, destination):
    """Preserve the input AO centres without replacing an older execution input."""
    text = bound_geometry_text(context)
    destination = Path(destination)
    if destination.exists() and destination.read_text() != text:
        raise ContractError(
            "Existing geometry differs from the bound Bohr coordinates; preserve its execution directory"
        )
    destination.write_text(text)


def context_from_files(
    case,
    basis_file,
    checkpoint,
    config,
    *,
    electrons=16,
    role_reference=None,
    reference_context=None,
    input_partition=None,
    role_provenance=None,
    required_active=None,
):
    config = configuration(config)
    states = _states(case["states"])
    basis_file, checkpoint = Path(basis_file).resolve(), Path(checkpoint).resolve()
    mo = read_openmolcas_orbital_hdf5(basis_file)
    inp = read_inporb(checkpoint)
    if len(mo.symmetry_block_sizes) != 1:
        raise ContractError("Native controller currently requires NOSYM")
    if mo.type_indices != inp.type_indices or not np.allclose(
        mo.occupations, inp.occupations, atol=1e-6, rtol=0
    ):
        raise ContractError("HDF5 and orbital checkpoint differ")
    coeff, _ = read_rasorb_coefficients(checkpoint, symmetry_block_sizes=mo.symmetry_block_sizes)
    coeff = mo.native_coefficients_to_ao(coeff)
    if not np.allclose(coeff, mo.mo_coefficients, atol=1e-8, rtol=0):
        raise ContractError("HDF5 and checkpoint MO coefficients differ")
    if set(mo.type_indices) - {"f", "i", "2", "s", "d"}:
        raise ContractError("Full-CAS orbital partition required")
    if not np.allclose(
        mo.mo_coefficients.T @ mo.ao_overlap @ mo.mo_coefficients,
        np.eye(mo.orbital_count),
        atol=1e-7,
        rtol=0,
    ):
        raise ContractError("MO basis is not AO-metric orthonormal")
    _geometry(case, mo, basis_file)
    parts = {
        name: tuple(i for i, t in enumerate(mo.type_indices) if t == tag)
        for name, tag in [("active", "2"), ("inactive", "i"), ("frozen", "f"), ("deleted", "d")]
    }
    inherited_core = tuple(getattr(reference_context, "required_active", ()))
    if inherited_core:
        old_mo = read_openmolcas_orbital_hdf5(reference_context.payload["basis_file"])
        if not np.array_equal(mo.mo_coefficients, old_mo.mo_coefficients):
            raise ContractError(
                "Required active-core transport currently needs the unchanged full MO frame"
            )
        if required_active is not None and not set(inherited_core) <= set(required_active):
            raise ContractError("Required active core cannot be dropped during context transport")
    required_active = tuple(inherited_core if required_active is None else required_active)
    if inp.electron_count != electrons + 2 * (len(parts["inactive"]) + len(parts["frozen"])):
        raise ContractError("Checkpoint electron partition differs")
    composition = None
    composition_pair = None
    composition_channels = None
    composition_audit = None
    from .preferences import build_preferences

    preferences = build_preferences(mo, config.get("composition_preferences"))
    if "composition" in config["channels"]:
        composition = ao_subspace_projection_scores(
            mo.mo_coefficients, mo.ao_overlap, select_ao_indices(mo, case["target"]["rules"])
        ).tolist()
        if config["composition_model"] in metrics.PAIR_COMPOSITION_MODELS:
            pair, composition_audit = determinant_pair_matrix(mo, case["target"]["rules"])
            composition_pair = pair.tolist()
        elif config["composition_model"] == "harmonic_channel_pair_v1":
            composition_channels, composition_audit = harmonic_channel_data(
                mo, case["target"]["rules"]
            )
    protocol = {
        key: config[key]
        for key in (
            "backend",
            "max_macro",
            "max_super_ci",
            "thr_energy",
            "thr_orbital",
            "thr_ci",
            "orbital_optimizer",
            "dmrg_probe_m",
            "dmrg_optimize_m",
            "dmrg_max_sweeps",
        )
    }
    if config.get("orbital_optimization") is False:
        protocol["orbital_optimization"] = False
    if config["backend"] == "dmrg":
        protocol["terminal_gauge"] = (
            "fixed_input_MO_basis_v1"
            if config.get("orbital_optimization") is False
            else "SA_natural_descending_AO_pivot_v1"
            if "qit" in config["channels"]
            else "native_optimizer_output_basis_v1"
        )
    context = Context(
        basis_id=file_hash(basis_file),
        geometry_id=identity(case["geometry_xyz"]),
        backend=config["backend"],
        nmo=mo.orbital_count,
        total_electrons=inp.electron_count,
        electrons=electrons,
        **parts,
        composition=composition,
        composition_pair=composition_pair,
        composition_channels=composition_channels,
        composition_preferences=preferences,
        states=states,
        protocol_id=identity(protocol),
        effective_config_id=identity(config),
        required_active=required_active,
        payload=dict(
            case=copy.deepcopy(case),
            basis_file=str(basis_file),
            checkpoint=str(checkpoint),
            checkpoint_hash=file_hash(checkpoint),
            protocol=protocol,
            gauge="native_optimizer_output_basis_v1"
            if config["backend"] == "dmrg"
            else "native_SA_natural_orbitals_v1",
        ),
    )
    if composition_audit is not None:
        context.payload["composition_audit"] = composition_audit
    from . import electron_roles

    if sum(x is not None for x in (role_reference, reference_context, role_provenance)) > 1:
        raise ContractError("Specify exactly one electron-role source")
    try:
        if role_reference is not None:
            roles, proof = electron_roles.from_reference(
                role_reference, binding(basis_file), electron_roles.partition(context)
            )
        elif reference_context is not None:
            if input_partition is None:
                raise ContractError("Role transport requires the actual input partition")
            roles, proof = electron_roles.from_context(
                reference_context,
                input_partition,
                binding(basis_file),
                electron_roles.partition(context),
            )
        elif role_provenance is not None:
            proof = copy.deepcopy(role_provenance)
            roles = electron_roles.replay(
                proof, binding(basis_file), electron_roles.partition(context)
            )
        else:
            return context  # A parsed, uninitialized container cannot execute.
        context.electron_roles = tuple(roles)
        context.payload["electron_role_provenance"] = proof
        context.__post_init__()
    except ValueError as error:
        raise ContractError(str(error)) from error
    return context


def _check_context(context, config):
    p = context.payload
    if (
        file_hash(p["basis_file"]) != context.basis_id
        or file_hash(p["checkpoint"]) != p["checkpoint_hash"]
    ):
        raise ContractError("Context physical files changed")
    if config["backend"] != context.backend:
        raise ContractError("Backend changed across a context")
    from . import electron_roles

    if not context.electron_roles:
        raise ContractError("Missing SCF electron-role provenance for physical execution")
    try:
        electron_roles.verify_context(context)
    except ValueError as error:
        raise ContractError(str(error)) from error
    current = context_from_files(
        p["case"],
        p["basis_file"],
        p["checkpoint"],
        config,
        electrons=context.electrons,
        role_provenance=p["electron_role_provenance"],
        required_active=context.required_active,
    )
    current.stable_order = context.stable_order
    if current.context_id != context.context_id:
        raise ContractError("Physical context or frozen protocol changed")


def _audit_output(previous, output, partition, kind):
    from scipy.optimize import linear_sum_assignment

    for field in (
        "geometry_id",
        "backend",
        "nmo",
        "total_electrons",
        "electrons",
        "states",
        "protocol_id",
        "effective_config_id",
        "required_active",
    ):
        if getattr(previous, field) != getattr(output, field):
            raise ContractError("Solver changed physical problem: " + field)
    for name in ("active", "inactive", "frozen", "deleted"):
        if len(getattr(output, name)) != len(partition[name]):
            raise ContractError("Solver changed partition dimensions: " + name)
    old = read_openmolcas_orbital_hdf5(previous.payload["basis_file"])
    new = read_openmolcas_orbital_hdf5(output.payload["basis_file"])
    with h5py.File(output.payload["basis_file"]) as h:
        if int(h.attrs["NACTEL"]) != partition["electrons"]:
            raise ContractError("Solver HDF5 active electron count differs")
    if (
        old.basis_function_ids.shape != new.basis_function_ids.shape
        or not np.array_equal(old.basis_function_ids, new.basis_function_ids)
        or old.center_labels != new.center_labels
        or not np.allclose(old.ao_overlap, new.ao_overlap, atol=1e-8, rtol=0)
    ):
        raise ContractError("Solver changed AO basis/metric")
    cross = old.mo_coefficients.T @ old.ao_overlap @ new.mo_coefficients
    for name in ("frozen", "deleted"):
        before = list(getattr(previous, name))
        after = list(getattr(output, name))
        if (
            before
            and np.min(np.linalg.svd(cross[np.ix_(before, after)], compute_uv=False)) < 1 - 1e-7
        ):
            raise ContractError("Solver changed protected physical subspace: " + name)
    if kind == "probe":
        for before, after in [
            (partition["active"], output.active),
            (partition["inactive"], output.inactive),
        ]:
            if len(before) != len(after):
                raise ContractError("Fixed-MO calculation changed partition dimensions")
            if (
                before
                and np.min(np.linalg.svd(cross[np.ix_(before, after)], compute_uv=False)) < 1 - 1e-7
            ):
                raise ContractError("Fixed-MO calculation changed a physical partition subspace")
        if previous.backend == "cas":
            # Pinned CIONLY preserves the CI basis, including non-natural D.
            # RDVEC/VecSort only groups types stably; a projector match alone
            # would miss an unreported rotation or permutation within active.
            groups = [
                ("frozen", "f"),
                ("inactive", "i"),
                ("active", "2"),
                ("secondary", "s"),
                ("deleted", "d"),
            ]
            order = [i for name, _ in groups for i in sorted(partition[name])]
            tags = tuple(tag for name, tag in groups for _ in partition[name])
            if new.type_indices != tags or not np.allclose(
                new.mo_coefficients, old.mo_coefficients[:, order], atol=1e-8, rtol=0
            ):
                raise ContractError(
                    "Fixed CASCI output MO coefficients/order differ from the requested input basis"
                )
    rows, cols = linear_sum_assignment(-(np.abs(cross) ** 2))
    mapping = dict(zip(rows.tolist(), cols.tolist()))
    output.stable_order = tuple(mapping[i] for i in previous.stable_order)
    return dict(
        stable_order_transport="maximum_AO_metric_overlap_assignment",
        old_to_new=[mapping[i] for i in range(previous.nmo)],
    )


def canonical_natural_rotation(density, coefficients, overlap, *, degeneracy_tolerance=1e-10):
    """Descending occupations; degenerate eigenspaces pinned to ordered AO axes.

    The target atom/AO projector is deliberately not used. Within a cluster of
    occupations differing by at most 1e-10, projected AO axes are orthogonalized
    in their fixed physical AO order. Largest-coefficient phase fixes each MO.
    """
    values, rotation = np.linalg.eigh(density)
    order = np.argsort(-values, kind="stable")
    values = values[order]
    rotation = rotation[:, order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[start] - values[end] <= degeneracy_tolerance:
            end += 1
        if end - start > 1:
            block = coefficients @ rotation[:, start:end]
            projected = block.T @ overlap
            vectors = []
            for axis in projected.T:
                vector = axis.copy()
                for _ in range(2):
                    for prior in vectors:
                        vector -= prior @ vector * prior
                norm = np.linalg.norm(vector)
                if norm > 1e-10:
                    vectors.append(vector / norm)
                if len(vectors) == end - start:
                    break
            if len(vectors) != end - start:
                raise ContractError("Degenerate natural subspace lacks stable AO pivots")
            rotation[:, start:end] = rotation[:, start:end] @ np.stack(vectors, axis=1)
        start = end
    final = coefficients @ rotation
    for i in range(final.shape[1]):
        pivot = np.flatnonzero(np.abs(final[:, i]) >= np.max(np.abs(final[:, i])) - 1e-12)[0]
        if final[pivot, i] < 0:
            rotation[:, i] *= -1
    if not np.allclose(rotation.T @ rotation, np.eye(len(values)), atol=1e-10, rtol=0):
        raise ContractError("Natural gauge rotation is not orthogonal")
    return rotation


def _replace_orbital_section(text, name, content):
    pattern = re.compile(r"^\s*#" + name + r"\s*$.*?(?=^\s*#[A-Za-z]|\Z)", re.M | re.S)
    if not pattern.search(text):
        raise ContractError("Missing checkpoint #" + name + " section")
    return pattern.sub(lambda _: "#" + name + "\n" + content + "\n", text, count=1)


def verify_native_inporb22(path, expected_coefficients):
    """Verify RDVEC fixed records independently of our whitespace parser."""
    text = Path(path).read_text()
    lines = text.splitlines()
    if not lines or lines[0].strip() != "#INPORB 2.2":
        raise ContractError("Native fixed-MO export requires verified INPORB 2.2 records")
    expected = np.asarray(expected_coefficients)
    nbf, nmo = expected.shape
    try:
        index = lines.index("#ORB") + 1
    except ValueError as error:
        raise ContractError("Missing native orbital record section") from error
    actual = np.empty_like(expected)
    for orbital in range(nmo):
        if index >= len(lines) or not re.fullmatch(
            r"\s*\*\s+ORBITAL\s+1\s+" + str(orbital + 1) + r"\s*", lines[index]
        ):
            raise ContractError("Native fixed-width orbital record boundary differs")
        index += 1
        values = []
        for start in range(0, nbf, 5):
            if index >= len(lines):
                raise ContractError("Truncated native orbital coefficients")
            line = lines[index]
            index += 1
            count = min(5, nbf - start)
            if len(line) < 22 * count or line[22 * count :].strip():
                raise ContractError("Native orbital record has incorrect fixed-width fields")
            try:
                values.extend(
                    float(line[22 * i : 22 * (i + 1)].replace("D", "E")) for i in range(count)
                )
            except ValueError as error:
                raise ContractError("Blank or malformed native coefficient field") from error
        actual[:, orbital] = values
    while index < len(lines) and not lines[index].strip():
        index += 1
    if index >= len(lines) or not lines[index].startswith("#"):
        raise ContractError("Unexpected trailing native orbital records")
    if not np.allclose(actual, expected, atol=1e-12, rtol=1e-12):
        raise ContractError("Native fixed-width interpretation changes the supplied orbital basis")
    return dict(
        format="INPORB_2.2_5x22_RDVEC_fixed_records",
        coefficient_max_error=float(abs(actual - expected).max()),
    )


def read_dmrg_densities(context):
    """Full per-root D in the native output MO basis, including zero-HDF5 cases.

    The pinned external-CI build allocates but does not populate DENSITY_MATRIX.
    Its per-root natural-orbital files contain the exact spectral decomposition
    of each root density. Rotate that complete decomposition into the output basis;
    never substitute just its diagonal or normalize a zero/incorrect density.
    """
    path = Path(context.payload["basis_file"])
    mo = read_openmolcas_orbital_hdf5(path)
    _energy_check(path, context.states)
    with h5py.File(path) as h:
        raw = np.asarray(h["DENSITY_MATRIX"][()], float) if "DENSITY_MATRIX" in h else None
        energies = np.asarray(h["ROOT_ENERGIES"][()], float).reshape(-1)
        derived_fixed = (
            h.attrs.get("PAS_ORBITAL_CONTAINER_ROLE")
            == "derived_fixed_MO_partition_and_final_MPS_density"
        )
    if raw is not None and np.any(raw):
        d, _ = read_densities(
            path, context.states, context.electrons, require_ci=False, require_natural=False
        )
        if not np.allclose(
            np.diag(np.einsum("r,rij->ij", context.states["weights"], d)),
            mo.occupations[list(context.active)],
            atol=1e-6,
            rtol=0,
        ):
            raise ContractError("Native root densities differ from output diagonal occupations")
        return (
            d,
            energies,
            dict(
                method="derived_fixed_MPS_full_root_densities"
                if derived_fixed
                else "native_HDF5_full_root_densities",
                sources=dict(basis=binding(path)),
            ),
        )
    active = list(context.active)
    reference = mo.mo_coefficients[:, active]
    matrices = []
    root_energies = []
    sources = dict(basis=binding(path))
    for root in context.states["roots"]:
        orbital = Path(context.payload["checkpoint"] + "." + str(root))
        if not orbital.is_file():
            raise ContractError(
                "Native DMRG HDF5 density unavailable and root natural orbitals missing"
            )
        titles = re.findall(
            r"RASSCF natural orbitals for root number\s+(\d+)\s+E=\s*([-+\d.EeDd]+)",
            orbital.read_text(),
        )
        if len(titles) != 1 or int(titles[0][0]) != root:
            raise ContractError("Root natural orbital state/energy title is missing or differs")
        root_energies.append(float(titles[0][1].replace("D", "E").replace("d", "e")))
        inp = read_inporb(orbital)
        coeff, tags = read_rasorb_coefficients(
            orbital, symmetry_block_sizes=mo.symmetry_block_sizes
        )
        coeff = mo.native_coefficients_to_ao(coeff)
        selected = [i for i, t in enumerate(tags) if t == "2"]
        if len(selected) != context.k or inp.electron_count != context.total_electrons:
            raise ContractError("Root natural orbital electron/active partition differs")
        if not np.allclose(coeff.T @ mo.ao_overlap @ coeff, np.eye(context.nmo), atol=1e-7, rtol=0):
            raise ContractError("Root natural orbitals are not AO-metric orthonormal")
        rotation = reference.T @ mo.ao_overlap @ coeff[:, selected]
        if not np.allclose(rotation.T @ rotation, np.eye(context.k), atol=1e-7, rtol=0):
            raise ContractError("Root natural active subspace differs from final output basis")
        matrices.append((rotation * np.asarray(inp.occupations)[selected]) @ rotation.T)
        sources["root_" + str(root)] = binding(orbital)
    d = np.asarray(matrices)
    metrics.validate_densities(d, context.states["weights"], context.electrons)
    averaged = np.einsum("r,rij->ij", context.states["weights"], d)
    root_energies = np.asarray(root_energies)
    if not np.all(np.isfinite(root_energies)) or not np.allclose(
        root_energies, energies, atol=1e-7, rtol=0
    ):
        raise ContractError("Root natural orbital energies differ from the converged native output")
    calls = sorted(path.parent.glob("block2_calls/call_*/energy_density.json"))
    if calls:
        final_energy = read_json(calls[-1])
        expected = np.asarray([s["energy"] for s in final_energy["states"]], float)
        if expected.shape != root_energies.shape or not np.allclose(
            root_energies, expected, atol=1e-10, rtol=0
        ):
            raise ContractError(
                "Root natural orbital energies differ from the final audited density"
            )
        sources["final_gradient_density"] = binding(calls[-1])
    # Molcas writes average MO occupations/HDF5 energies before its final CI
    # evaluation; per-root natural orbitals use that later density. Their small
    # difference is not a density reconstruction error. Keep the full matrix,
    # validate the root energy binding, and report the cross-stage discrepancy.
    return (
        d,
        root_energies,
        dict(
            method="exact_root_natural_spectral_decomposition_in_output_MO_basis",
            sources=sources,
            average_occupations_role="approximate_earlier_native_output_not_final_density_constraint",
            earlier_native_diagonal_max_difference=float(
                np.max(abs(np.diag(averaged) - mo.occupations[active]))
            ),
            final_root_minus_native_energies=(root_energies - energies).tolist(),
            averaged_offdiagonal_norm=float(np.linalg.norm(averaged - np.diag(np.diag(averaged)))),
            original_HDF5_density_status="zero_placeholder" if raw is not None else "absent",
        ),
    )


def canonicalize_dmrg_context(context, directory, config):
    """Create an actual deterministic final gauge before fresh MPS evaluation.

    Only active rotations occur; this is evidence normalization, not a search
    action. Original optimizer artifacts remain intact. Stale CI vectors are
    removed from the derived orbital container rather than mislabelled.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    old = context.payload
    mo = read_openmolcas_orbital_hdf5(old["basis_file"])
    densities, root_energies, density_source = read_dmrg_densities(context)
    averaged = np.einsum("r,rij->ij", context.states["weights"], densities)
    active = list(context.active)
    rotation = canonical_natural_rotation(averaged, mo.mo_coefficients[:, active], mo.ao_overlap)
    full = np.eye(context.nmo)
    full[np.ix_(active, active)] = rotation
    native, _ = read_rasorb_coefficients(
        old["checkpoint"], symmetry_block_sizes=mo.symmetry_block_sizes
    )
    native = native @ full
    transformed = np.einsum("ia,rij,jb->rab", rotation, densities, rotation)
    occupations = mo.occupations.copy()
    occupations[active] = np.diag(rotation.T @ averaged @ rotation)
    energies = np.diag(full.T @ np.diag(mo.energies) @ full)
    checkpoint = directory / "canonical.RasOrb"
    basis = directory / "canonical.rasscf.h5"

    # RDVEC uses fixed-width records, not whitespace-separated free input.
    # INPORB 2.2 requires 5(1X,ES21.14) for ORB/OCC, 10(1X,ES11.4)
    # for ONE. One coefficient per line silently supplies zero blank fields.
    def records(values, columns, width, precision):
        return "".join(
            "".join(
                " " + format(float(v), f"{width - 1}.{precision}E")
                for v in values[start : start + columns]
            )
            + "\n"
            for start in range(0, len(values), columns)
        )

    orb = "".join(
        f"* ORBITAL 1 {i + 1}\n" + records(native[:, i], 5, 22, 14) for i in range(context.nmo)
    )
    text = re.sub(r"^#INPORB\s+[^\n]+", "#INPORB 2.2", Path(old["checkpoint"]).read_text(), count=1)
    text = _replace_orbital_section(text, "ORB", orb)
    text = _replace_orbital_section(
        text, "OCC", "* OCCUPATION NUMBERS\n" + records(occupations, 5, 22, 14)
    )
    if re.search(r"^\s*#ONE\s*$", text, re.M):
        text = _replace_orbital_section(
            text, "ONE", "* ONE ELECTRON ENERGIES\n" + records(energies, 10, 12, 4)
        )
    checkpoint.write_text(text)
    fixed_record_proof = verify_native_inporb22(checkpoint, native)
    shutil.copy2(old["basis_file"], basis)
    with h5py.File(basis, "r+") as h:
        h["ROOT_ENERGIES"][...] = root_energies.reshape(h["ROOT_ENERGIES"].shape)
        h["MO_VECTORS"][...] = native.ravel(order="F").reshape(h["MO_VECTORS"].shape)
        h["MO_OCCUPATIONS"][...] = occupations
        h["MO_ENERGIES"][...] = energies
        if "DENSITY_MATRIX" in h:
            h["DENSITY_MATRIX"][...] = transformed.reshape(h["DENSITY_MATRIX"].shape)
        else:
            h.create_dataset("DENSITY_MATRIX", data=transformed)
        if "SPINDENSITY_MATRIX" in h:
            d = np.asarray(h["SPINDENSITY_MATRIX"][()], float)
            if d.shape not in (densities.shape, (len(densities), context.k * context.k)):
                raise ContractError("Unsupported spin-density layout in gauge conversion")
            h["SPINDENSITY_MATRIX"][...] = np.einsum(
                "ia,rij,jb->rab", rotation, d.reshape(densities.shape), rotation
            ).reshape(d.shape)
        for name in ("CI_VECTORS", "TRANSITION_DENSITY_MATRIX", "TRANSITION_SPIN_DENSITY_MATRIX"):
            if name in h:
                del h[name]
        h.attrs["PAS_ORBITAL_CONTAINER_ROLE"] = "derived_final_gauge_requires_fresh_MPS_evidence"
    output = context_from_files(
        old["case"],
        basis,
        checkpoint,
        config,
        electrons=context.electrons,
        reference_context=context,
        input_partition=context.partition(context.active),
    )
    transport = _audit_output(context, output, context.partition(context.active), "probe")
    gauge = dict(
        name="SA_natural_descending_AO_pivot_v1",
        occupation_cluster_tolerance=1e-10,
        phase="largest_absolute_AO_coefficient_positive_first_tie",
        degenerate_basis="ordered_AO_projection_twopass_Gram_Schmidt",
        source_basis=binding(old["basis_file"]),
        source_checkpoint=binding(old["checkpoint"]),
        density_source=density_source,
        native_fixed_record_proof=fixed_record_proof,
        active_rotation=rotation.tolist(),
        basis_mapping=transport,
        density_role="optimizer_final_state_average_for_gauge_only",
        fresh_fixed_MPS_required=True,
    )
    output.payload["gauge"] = gauge["name"]
    write_json(directory / "gauge.json", gauge)
    return output, binding(directory / "gauge.json")


def write_partition(context, active, destination, *, survey=False):
    """Write requested types exactly; no occupation-role selection or retention."""
    if survey:
        from .diagnostic_window import survey_partition

        parts = survey_partition(context, active)
    else:
        parts = context.partition(active)
    tags = [""] * context.nmo
    for key, tag in [
        ("active", "2"),
        ("inactive", "i"),
        ("frozen", "f"),
        ("secondary", "s"),
        ("deleted", "d"),
    ]:
        for i in parts[key]:
            if tags[i]:
                raise ContractError("Overlapping requested partitions")
            tags[i] = tag
    if any(not t for t in tags):
        raise ContractError("Incomplete requested partition")
    source = Path(context.payload["checkpoint"])
    text = _replace_type_indices(source.read_text(), tags)
    destination = Path(destination)
    destination.write_text(text)
    if list(read_inporb(destination).type_indices) != tags:
        raise ContractError("Orbital type write roundtrip failed")
    return parts


def gateway_input(case, geometry_text=None):
    """Use an explicitly bound basis library when the standard name is insufficient."""
    library = case.get("basis_library")
    by_element = case.get("basis_by_element")
    cholesky = case.get("cholesky")
    if cholesky is not None and cholesky is not False:
        raise ContractError("Only explicit conventional integral mode cholesky=False is supported")
    integral_options = "NOCD\n" if cholesky is False else ""
    if by_element is not None:
        if library is not None:
            raise ContractError("Specify a native basis map or an inline library, not both")
        if not isinstance(by_element, dict) or not by_element:
            raise ContractError("Missing per-element basis map")
        lines = (geometry_text or case["geometry_xyz"]).splitlines()
        atoms = [line.split() for line in lines[2:] if line.strip()]
        if len(atoms) != int(lines[0]) or any(len(atom) != 4 for atom in atoms):
            raise ContractError("Malformed per-element basis geometry")
        symbols = {atom[0].capitalize() for atom in atoms}
        if set(by_element) != symbols:
            raise ContractError("Per-element basis map must match the geometry elements")
        units = "" if lines[1].strip().lower() == "bohr" else " /angstrom"
        text = "&GATEWAY\n" + integral_options
        for i, (symbol, *xyz) in enumerate(atoms, 1):
            symbol = symbol.capitalize()
            label = by_element[symbol]
            if (
                not isinstance(label, str)
                or re.fullmatch(r"[A-Za-z0-9_.+-]+", label) is None
                or label.split(".")[0].capitalize() != symbol
                or not np.isfinite(np.asarray(xyz, float)).all()
            ):
                raise ContractError("Invalid native basis label or coordinate")
            text += f"Basis set\n{label}\n{symbol}{i} {' '.join(xyz)}{units}\nEnd of basis\n"
        return text
    if library is None:
        return (
            f"&GATEWAY\n{integral_options}Coord=geometry.xyz\nBasis={case['basis']}\nGroup=NOSYM\n"
        )
    blocks = {}
    for entry in verify_binding(library).read_text().split("\n/"):
        lines = entry.lstrip("/").strip().splitlines()
        if len(lines) < 4 or lines[0] in blocks:
            raise ContractError("Malformed or duplicate inline basis entry")
        # Molcas library entries contain two comment lines before the numeric data.
        blocks[lines[0]] = "\n".join(lines[3:])
    lines = (geometry_text or case["geometry_xyz"]).splitlines()
    atoms = [line.split() for line in lines[2:] if line.strip()]
    if len(atoms) != int(lines[0]) or any(len(atom) != 4 for atom in atoms):
        raise ContractError("Malformed inline-basis geometry")
    units = "" if lines[1].strip().lower() == "bohr" else " /angstrom"
    text = "&GATEWAY\nExpert\n" + integral_options
    for i, (symbol, *xyz) in enumerate(atoms, 1):
        symbol = symbol.capitalize()  # Native H5 labels use TI/CL; libraries use Ti/Cl.
        label = library["labels"][symbol]
        if not np.isfinite(np.asarray(xyz, float)).all() or label not in blocks:
            raise ContractError("Invalid inline basis/coordinate binding")
        text += f"Basis set\n{label} / Inline\n{blocks[label]}\n{symbol}{i} {' '.join(xyz)}{units}\nEnd of basis\n"
    # Explicit atom blocks with no symmetry generators define C1. GROUP belongs
    # to the separate COORD/XYZ input mode and cannot accompany these blocks.
    return text


def seward_input(case, backend):
    rel = case["relativistic"]
    if str(rel).lower() in ("none", "nonrelativistic"):
        return "&SEWARD\n"
    return "&SEWARD\n" + (rel if backend == "dmrg" else "Relativistic=" + rel) + "\n"


def input_text(context, partition, kind, config):
    case = context.payload["case"]
    states = context.states
    n = len(states["roots"])
    # Molcas CIROOT reads integer relative weights, unlike the normalized
    # floating weights in HDF5 and Block2. Round-trip the rational ratio.
    fractions = [Fraction(str(w)).limit_denominator(1_000_000) for w in states["weights"]]
    denominator = lcm(*(w.denominator for w in fractions))
    integers = [w.numerator * (denominator // w.denominator) for w in fractions]
    divisor = reduce(gcd, integers)
    integers = [w // divisor for w in integers]
    if max(integers) > 2_147_483_647 or not np.allclose(
        np.array(integers) / sum(integers), states["weights"], atol=1e-12, rtol=0
    ):
        raise ContractError("Root weights cannot be represented by native CIROOT integer ratios")
    rootline = (
        f"CIroot={n},{n}; {' '.join(map(str, states['roots']))}; {' '.join(map(str, integers))}"
    )
    text = (
        gateway_input(
            case,
            bound_geometry_text(context)
            if "basis_library" in case or "basis_by_element" in case
            else None,
        )
        + seward_input(case, context.backend)
        + ">> COPY $CurrDir/trial.RasOrb INPORB\n&RASSCF\nLumOrb\nTypeIndex\n"
        + f"Spin={states['spin_multiplicity']}\nNACTEL={partition['electrons']} 0 0\n{rootline}\n"
        f"ITERations={config['max_macro']},{config['max_super_ci']}\nTHRS\n"
        f"{config['thr_energy']:.12g} {config['thr_orbital']:.12g} {config['thr_ci']:.12g}\n"
    )
    if kind != "optimize" or config.get("orbital_optimization") is False:
        text += "CIONLY\n"
    if context.backend == "dmrg":
        text += f"CISolver=BLOCK\nDMRG={config['dmrg_optimize_m'] if kind == 'optimize' else config['dmrg_probe_m']}\n"
    return text


def run_solver(directory, text, threads, *, environment=None):
    _allow_dispatch()
    import resource

    directory = checked_directory(directory, threads)
    (directory / "work").mkdir(exist_ok=True)
    (directory / "job.input").write_text(text)
    env = dict(os.environ)
    env.update(
        MOLCAS=CAS_MOLCAS,
        MOLCAS_NPROCS="1",
        MOLCAS_THREADS=str(threads),
        OMP_NUM_THREADS=str(threads),
        MKL_NUM_THREADS=str(threads),
        OPENBLAS_NUM_THREADS=str(threads),
        MOLCAS_MEM="16000",
        MOLCAS_WORKDIR=str(directory / "work"),
        MOLCAS_PROJECT="job",
        MOLCAS_OUTPUT=str(directory),
        PMIX_MCA_pcompress_base_silence_warning="1",
    )
    libs = [
        "/nfs-soft/boost_1_86_0/v1/lib",
        "/nfs-soft/gsl-2.8/v1/lib",
        "/nfs-soft/hdf5-git-1.14.5/v2/lib",
        "/opt/intel/oneapi/mkl/2025.1/lib",
        "/nfs-soft/openmpi-5.0.6/v1/lib",
    ]
    env["LD_LIBRARY_PATH"] = ":".join(libs + [env.get("LD_LIBRARY_PATH", "")])
    env.update(environment or {})
    started = directory / "started.json"
    if started.exists():
        raise RuntimeError("Unreconciled physical start; automatic rerun is forbidden")
    started_record = dict(pid=os.getpid(), started_unix=time.time())
    write_json(started, started_record)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    with (directory / "solver.log").open("w") as log:
        job = subprocess.Popen(
            [str(Path(env["MOLCAS"]) / "pymolcas"), "-np", "1", "-nt", str(threads), "job.input"],
            cwd=directory,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        process_stat = Path(f"/proc/{job.pid}/stat").read_text().rsplit(")", 1)[1].split()
        write_json(
            started, dict(started_record, solver_pid=job.pid, solver_start_ticks=process_stat[19])
        )
        job.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    for name in (
        "job.rasscf.h5",
        "job.RasOrb",
        "job.RasOrb.1",
        "job.RasOrb.2",
        "job.scf.h5",
        "job.ScfOrb",
    ):
        source = directory / "work/job" / name
        target = directory / name
        if not target.exists() and source.exists():
            shutil.copy2(source, target)
    result = dict(
        return_code=job.returncode,
        cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        wall_seconds=time.perf_counter() - start,
        threads=threads,
        memory_mb=16000,
        physical_runtime=dict(
            pymolcas=binding(Path(env["MOLCAS"]) / "pymolcas"),
            molcas_root=env["MOLCAS"],
            python=sys.executable,
        ),
    )
    write_json(directory / "execution.json", result)
    if env.get("PAS_RELEASE_AO_INTEGRALS") == "1":
        # pymolcas and its synchronous modules have all exited; subsequent MPS
        # analysis consumes FCIDUMP, never this regenerable AO integral file.
        integral = directory / "work/job/job.OrdInt"
        if integral.exists():
            if (
                integral.is_symlink()
                or not integral.resolve().is_relative_to(directory / "work")
                or not integral.is_file()
                or integral.stat().st_uid != os.getuid()
            ):
                raise ContractError("Unsafe completed AO-integral cleanup path")
            stat = integral.stat()
            cleanup = dict(
                path=str(integral),
                bytes=stat.st_size,
                inode=stat.st_ino,
                input=binding(directory / "job.input"),
                execution=binding(directory / "execution.json"),
                solver_return_code=job.returncode,
                reason="completed_solver_regenerable_AO_integrals",
            )
            write_json(directory / "ao_cleanup_plan.json", cleanup)
            integral.unlink()
            write_json(
                directory / "ao_cleanup_completed.json",
                dict(cleanup, completed=True, time=time.time()),
            )
    return result


def _converged(log, kind, return_code):
    """Interpret the final module outcome using the pinned Molcas return codes."""
    stops = list(re.finditer(r"---\s*Stop Module:\s*rasscf\b[^\n]*/rc=(_RC_[A-Z_]+_)", log, re.I))
    final_code = stops[-1][1].upper() if stops else None
    starts = list(re.finditer(r"---\s*Start Module:\s*rasscf\b", log, re.I))
    terminal = log[starts[-1].start() :] if starts else log
    numerical = bool(
        re.search(
            r"No\s+convergence\s+after\s*\d+|RASSCF did not converge|not reach.*converg|not converged|convergence problem",
            terminal,
            re.I,
        )
    )
    # Both pinned installations define 96=NOT_CONVERGED and 36=NOT_AVAILABLE.
    # Exit 8 is not a native numerical-convergence outcome.
    if re.search(
        r"No space left|Permission denied|Segmentation fault|Traceback|command not found|cannot open|Input/output error",
        terminal,
        re.I,
    ):
        raise RuntimeError("Solver infrastructure/termination failure; inspect solver.log")
    if (
        return_code == 96
        and final_code in (None, "_RC_NOT_CONVERGED_")
        and (numerical or final_code == "_RC_NOT_CONVERGED_")
    ):
        return False
    if return_code != 0 or final_code != "_RC_ALL_IS_WELL_":
        raise RuntimeError("Solver infrastructure/termination failure; inspect solver.log")
    if kind == "optimize" and not re.search(
        r"^\s*Convergence\s+after\s*\d+\s+iterations\b", terminal, re.I | re.M
    ):
        raise ContractError("Optimizer termination lacks a positive or negative convergence result")
    return True


def verify_qn(log):
    """Require the affirmative native setting, not a keyword or table header.

    Already-converged orbitals need not perform an actual QN step, so the
    solver's enabled setting is the contract. Negated settings always fail.
    """
    if re.search(r"(?:do\s+not|don.t|no)\s+make\s+quasi[- ]?newton|NOQUne", log, re.I):
        raise ContractError("Solver explicitly disabled QN optimization")
    if not re.search(r"^\s*(?:\|\s*)?Make\s+Quasi[- ]?Newton\s+update\b", log, re.I | re.M):
        raise ContractError("Solver log lacks affirmative QN setting")
    return dict(enabled=True, verification="affirmative_native_Make_Quasi_Newton_update_setting")


def _check_dmrg_states(states, electrons, *, optimize_orbitals):
    states = _states(states)
    multiplicity = states["spin_multiplicity"]
    if (
        type(multiplicity) is not int
        or multiplicity < 1
        or electrons < multiplicity - 1
        or (electrons - multiplicity + 1) % 2
    ):
        raise ContractError("DMRG electron count and spin multiplicity differ")
    if optimize_orbitals and multiplicity != 1:
        raise ContractError(
            "Pinned DMRGSCF spatial-gradient adapter currently requires singlet states"
        )


def check_capabilities(context, kind, channels, config=None):
    """Reject unsupported work before input construction or a solver launch."""
    if (
        getattr(context, "required_active", ())
        and kind == "optimize"
        and (config or {}).get("orbital_optimization") is not False
    ):
        raise ContractError("Required active-core calculations currently need fixed orbitals")
    if context.backend == "cas" and "qit" in channels:
        raise ContractError("Native CAS does not supply basis-bound QIT")
    if kind in ("probe", "survey") and set(channels) == {"composition"}:
        raise ContractError("Fixed-MO composition is known; diagnostic adds no quality evidence")
    if context.backend == "dmrg":
        optimize_orbitals = (
            kind == "optimize" and (config or {}).get("orbital_optimization") is not False
        )
        _check_dmrg_states(context.states, context.electrons, optimize_orbitals=optimize_orbitals)
        if optimize_orbitals and (context.k > 64 or context.electrons < 2):
            raise ContractError(
                "Pinned DMRGSCF gradient adapter requires 2+ electrons and <=64 target orbitals"
            )


def check_initial_capabilities(case, config):
    """Validate the declared initialization before spending even the SCF cost."""
    states = _states(case["states"])
    if config["backend"] == "dmrg":
        active = case["active_space"]
        optimize_orbitals = config.get("orbital_optimization") is not False
        _check_dmrg_states(states, active["electrons"], optimize_orbitals=optimize_orbitals)
        if optimize_orbitals and (active["orbitals"] > 64 or active["electrons"] < 2):
            raise ContractError(
                "Pinned DMRGSCF gradient adapter requires 2+ electrons and <=64 target orbitals"
            )


def _energy_check(path, states, expected=None):
    with h5py.File(path) as h:
        a = h.attrs
        if (
            int(a["SPINMULT"]) != states["spin_multiplicity"]
            or np.asarray(a["STATE_ROOTID"]).reshape(-1).tolist() != states["roots"]
        ):
            raise ContractError("Returned HDF5 root/spin identity differs")
        if "STATE_WEIGHT" in a and not np.allclose(
            np.asarray(a["STATE_WEIGHT"]).reshape(-1), states["weights"], atol=1e-12, rtol=0
        ):
            raise ContractError("Returned state weights differ")
        energies = np.asarray(h["ROOT_ENERGIES"][()], float).reshape(-1)
    if energies.shape != (len(states["roots"]),) or not np.all(np.isfinite(energies)):
        raise ContractError("Returned root energies are missing/nonfinite")
    energy = float(energies @ states["weights"])
    if expected is not None and not np.isclose(energy, expected, atol=1e-7, rtol=0):
        raise ContractError("Imported incumbent and HDF5 state-averaged energy differ")
    return energy


def _copy_artifact(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        if file_hash(source) != file_hash(destination):
            raise ContractError("Initialization destination contains different artifacts")
    else:
        shutil.copy2(source, destination)
    return destination


class InitializationUnavailable(ContractError):
    """A required initial stage is absent; only verified predecessors may replace it."""


def _verify_seed_provenance(seed):
    """Recognize audited initial imports; never infer initialization from validity."""
    e = seed["incumbent"]
    s = seed["snapshot"]
    reuse = seed.get("casscf_reuse")
    if not all(e.get(k) for k in ("valid", "converged", "state_valid")):
        raise ContractError("Seed incumbent has no valid convergence/state receipt")
    if reuse is not None:
        if (
            reuse.get("schema") != "pcc_fock_frontier_casscf_seed_v1"
            or reuse.get("initialization_role") != "historical_initial_state_not_search_endpoint"
            or seed.get("fresh_scf") is not False
            or reuse.get("historical_search_results_reused") is not False
        ):
            raise ContractError("Invalid CASSCF cold-stage provenance")
        for record in reuse["inputs"].values():
            verify_binding(record)
        status = read_json(reuse["inputs"]["initial_status_artifact"]["path"])
    else:
        provenance = e.get("provenance", {})
        if provenance.get("convergence_source") != "archived_hash_bound_status_artifact":
            raise InitializationUnavailable("Seed lacks a hash-bound initial convergence status")
        directory = Path(s["basis_file"]).parent
        candidates = [
            p
            for p in directory.glob("*.json")
            if file_hash(p) == provenance.get("source_hashes", {}).get("initial_status_artifact")
        ]
        if not candidates:
            raise InitializationUnavailable(
                "Hash-bound cold-stage convergence status is unavailable"
            )
        if len(candidates) != 1:
            raise ContractError("Initial convergence status is ambiguous")
        status = read_json(candidates[0])
    if (
        status.get("schema") != "openmolcas_initial_evidence_status_v1"
        or status.get("scientific_convergence_verified") is not True
    ):
        raise InitializationUnavailable("Historical initialization convergence is not verified")
    if (
        status.get("hdf5_sha256") != s["basis_id"]
        or status.get("state_ids") != s["states"]["ids"]
        or not np.allclose(
            status.get("state_weights", []), s["states"]["weights"], atol=1e-12, rtol=0
        )
    ):
        raise ContractError(
            "Cold-stage convergence receipt does not bind this basis and state contract"
        )
    if (
        e.get("snapshot", {}).get("basis_id") != s["basis_id"]
        or e["snapshot"].get("checkpoint_hash") != s["checkpoint_hash"]
    ):
        raise ContractError("Initial incumbent and snapshot physical identities differ")
    return status


def initialize_fixed_dmrg(
    seed_path, config, output_directory, *, role_reference=None, fixed_dmrg_policy=None
):
    """Complete initial metrics with M1000 CI on the identical cold orbital basis.

    This explicit initialization action never optimizes or canonicalizes MOs,
    never becomes a probe/optimize learning event, and never falls back to SCF.
    A signed PAS cold seed supplies the orbitals, not the new density or energy.
    """
    import fcntl

    from . import electron_roles
    from .cold_start import physical_case

    config = configuration(config)
    if fixed_dmrg_policy is not None:
        from .solvers.block2_convergence import validate_fixed_policy

        fixed_dmrg_policy = validate_fixed_policy(fixed_dmrg_policy)
    if (
        config["backend"] != "dmrg"
        or set(config["channels"]) != {"occupation", "composition", "qit"}
        or config["dmrg_optimize_m"] != 1000
    ):
        raise ContractError("Fixed initial evidence requires three-channel DMRG at M1000")
    source = Path(seed_path).resolve()
    seed = read_json(source)
    if not {"context", "incumbent", "initialization"} <= set(seed):
        raise ContractError("Fixed initial evidence needs a signed PAS cold seed")
    old = Context.from_dict(seed["context"])
    initial = seed["initialization"]
    receipt = seed["incumbent"]
    if initial.get("role") not in (
        "new_physical_cold_stage",
        "imported_verified_cold_stage",
        "reused_frozen_cold_stage",
    ):
        raise ContractError("Source is not a recognized cold initialization")
    if (
        receipt.get("kind") != "initial"
        or not all(receipt.get(k) for k in ("valid", "converged", "state_valid"))
        or receipt.get("evidence_id")
        != identity({k: v for k, v in receipt.items() if k != "evidence_id"})
        or identity(receipt.get("output_context")) != identity(seed["context"])
    ):
        raise ContractError("Cold source receipt or context is invalid")
    for item in receipt.get("artifacts", {}).values():
        if isinstance(item, dict) and {"path", "sha256"} <= item.keys():
            verify_binding(item)
    sources = dict(
        basis_file=binding(old.payload["basis_file"]), checkpoint=binding(old.payload["checkpoint"])
    )
    if (
        sources["basis_file"]["sha256"] != old.basis_id
        or sources["checkpoint"]["sha256"] != old.payload["checkpoint_hash"]
        or any(receipt["artifacts"][key] != value for key, value in sources.items())
    ):
        raise ContractError("Cold orbital files do not match their signed source")
    if old.electron_roles:
        electron_roles.verify_context(old)
    role_args = (
        dict(role_provenance=old.payload["electron_role_provenance"])
        if old.electron_roles and role_reference is None
        else dict(role_reference=electron_roles.discover_reference(seed, role_reference))
    )
    source_case = physical_case(initial.get("source_case", old.payload["case"]))
    context = context_from_files(
        source_case,
        old.payload["basis_file"],
        old.payload["checkpoint"],
        config,
        electrons=old.electrons,
        **role_args,
    )
    context.stable_order = old.stable_order
    context.payload["gauge"] = "fixed_input_MO_basis_v1"
    if (
        context.basis_id != old.basis_id
        or context.active != old.active
        or context.states != old.states
        or context.total_electrons != old.total_electrons
    ):
        raise ContractError("Fixed initialization changed the cold physical space")
    partition = context.partition(context.active)
    _check_context(context, config)
    output = Path(output_directory).resolve()
    if output / "seed.json" == source:
        raise ContractError("Fixed initialization must not overwrite its source seed")
    output = checked_directory(output, config["threads"])
    request = dict(
        schema="pas_fixed_initial_evidence_request_v1",
        source_seed=binding(source),
        context_id=context.context_id,
        source_orbitals=sources,
        partition=partition,
        config=config,
        bond_dimension=1000,
        orbital_optimization=False,
    )
    if fixed_dmrg_policy is not None:
        request["fixed_dmrg_policy"] = fixed_dmrg_policy
    request["request_id"] = identity(request)
    with (output / "initialization.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request_file = output / "initial_request.json"
        if not request_file.exists() and any(
            (output / name).exists()
            for name in ("seed.json", "densities.npy", "fixed_evidence", "started.json")
        ):
            raise ContractError(
                "Fixed initialization destination already contains unbound artifacts"
            )
        if request_file.exists() and read_json(request_file) != request:
            raise ContractError("Fixed initialization request changed")
        if (output / "seed.json").exists():
            if not request_file.exists():
                raise ContractError("Completed fixed initialization lacks its request")
            return import_seed(output / "seed.json", config, output, allow_compute=False)
        started = output / "started.json"
        fixed = output / "fixed_evidence/fixed_result.json"
        if started.exists() and not fixed.exists():
            raise RuntimeError(
                "Unfinished fixed initialization; automatic solver rerun is forbidden"
            )
        write_json(request_file, request)
        if not started.exists():
            _allow_dispatch()
            write_json(
                started,
                dict(pid=os.getpid(), started_unix=time.time(), request_id=request["request_id"]),
            )
        runtime_identity = runtime()
        fixed_options = (
            {} if fixed_dmrg_policy is None else dict(fixed_dmrg_policy=fixed_dmrg_policy)
        )
        data = _fixed_dmrg(
            context,
            partition,
            context.payload["checkpoint"],
            output / "fixed_evidence",
            1000,
            config,
            **fixed_options,
        )
        # Recheck the immutable source even after failure, before issuing a receipt.
        for item in sources.values():
            verify_binding(item)
        if not data["valid"]:
            write_json(
                output / "initialization_failure.json",
                dict(
                    request_id=request["request_id"],
                    failure=data["failure"],
                    cpu_seconds=data["cpu_seconds"],
                    wall_seconds=data["wall_seconds"],
                ),
            )
            raise InitializationUnavailable(
                "Fixed initial DMRG evidence failed: " + data["failure"]
            )
        values = metrics.evaluate(
            config["channels"],
            config["weights"],
            densities=data["densities"],
            state_weights=context.states["weights"],
            electrons=context.electrons,
            composition=context.composition,
            active=context.active,
            entropies=data["entropies"],
            mutual_information=data["mutual_information"],
            composition_pair=context.composition_pair,
            composition_channels=context.composition_channels,
            composition_model=config["composition_model"],
            composition_preferences=context.composition_preferences,
            composition_purity_weight=config.get("composition_purity_weight", 0.5),
        )
        energies = np.asarray(data["root_energies"], float)
        if energies.shape != (len(context.states["roots"]),) or not np.isfinite(energies).all():
            raise ContractError("Fixed initial evidence lacks complete root energies")
        values["energy"] = float(energies @ context.states["weights"])
        np.save(output / "densities.npy", data["densities"])
        costs = {key: float(data[key]) for key in ("cpu_seconds", "wall_seconds")}
        if any(not np.isfinite(v) or v < 0 for v in costs.values()):
            raise ContractError("Invalid initialization cost")
        initialization = dict(
            role="new_physical_cold_stage",
            stage="fixed_M1000_evidence_on_existing_cold_basis",
            source_seed=binding(source),
            source_case=source_case,
            source_initialization=copy.deepcopy(initial),
            source_orbitals=sources,
            fresh_scf=False,
            solver_executions=1,
            orbital_optimization=False,
            orbital_basis_preserved=True,
            initial_selection_used_composition=initial.get("initial_selection_used_composition"),
            matches_search_protocol=False,
            search_learning_event=False,
            search_budgets_charged=False,
            generation_protocol=dict(
                method="fixed_MO_DMRGCI",
                bond_dimension=1000,
                convergence_profile="standard",
                all_metrics_from_same_fixed_MPS=True,
                composition_source="exact_AO_projection_in_unchanged_input_basis",
            ),
            historical_cpu_seconds=initial.get("historical_cpu_seconds"),
            historical_wall_seconds=initial.get("historical_wall_seconds"),
            historical_costs_unknown=initial.get("historical_costs_unknown", True),
            **costs,
        )
        for name in ("cpu_seconds", "wall_seconds"):
            historical = initial.get("known_historical_" + name, initial.get("historical_" + name))
            known = float(initial.get(name, 0.0)) + (historical or 0.0)
            initialization["known_historical_" + name] = known
            initialization["historical_" + name] = (
                None if initialization["historical_costs_unknown"] else known
            )
        artifacts = dict(
            **sources,
            densities=binding(output / "densities.npy"),
            mps_evidence=data["qit_artifact"],
            integrals=data["integral_artifact"],
            fixed_result=binding(fixed),
            request=binding(request_file),
        )
        completed = dict(
            request_id=request["request_id"],
            context_id=context.context_id,
            kind="initial",
            active=list(context.active),
            valid=True,
            converged=True,
            state_valid=True,
            metrics=values,
            output_context=context.to_dict(),
            artifacts=artifacts,
            provenance=initialization,
            runtime=runtime_identity,
            **costs,
        )
        if fixed_dmrg_policy is not None:
            initialization["generation_protocol"]["fixed_dmrg_policy"] = fixed_dmrg_policy
            completed["fixed_dmrg_policy"] = fixed_dmrg_policy
        completed["evidence_id"] = identity(completed)
        result = dict(context=context.to_dict(), incumbent=completed, initialization=initialization)
        write_json(output / "seed.json", result)
        return result


def import_seed(
    seed_path, config, output_directory, *, scf_source=None, allow_compute=True, role_reference=None
):
    config = configuration(config)
    source = Path(seed_path).resolve()
    seed = read_json(source)
    from .cold_start import from_verified_predecessor, physical_case

    def preceding(reason):
        if not allow_compute:
            raise InitializationUnavailable(
                reason
                + "; run init explicitly to complete initialization before frozen-package import"
            )
        return from_verified_predecessor(seed, source, config, output_directory, reason, scf_source)

    if "context" in seed:
        context = Context.from_dict(seed["context"])
        initial = seed["initialization"]
        receipt = seed["incumbent"]
        if initial.get("role") not in (
            "new_physical_cold_stage",
            "imported_verified_cold_stage",
            "reused_frozen_cold_stage",
        ):
            raise ContractError("PAS package is not a recognized initialization stage")
        if (
            "composition" not in config["channels"]
            and initial.get("initial_selection_used_composition") is not False
        ):
            return preceding("Initial selection used composition or has unknown dependencies")
        if not all(receipt.get(key) for key in ("valid", "converged", "state_valid")):
            return preceding("PAS seed lacks a valid initial physical result")
        unsigned = {key: value for key, value in receipt.items() if key != "evidence_id"}
        if receipt.get("evidence_id") != identity(unsigned):
            raise ContractError("PAS seed physical receipt changed")
        # Verify the signed serialized schema before adding current Context
        # defaults; a v01 cold package legitimately lacks new v02 fields.
        if identity(receipt.get("output_context")) != identity(seed["context"]):
            raise ContractError("PAS incumbent output context differs from seed")
        for item in receipt.get("artifacts", {}).values():
            if isinstance(item, dict) and {"path", "sha256"} <= item.keys():
                verify_binding(item)
        if (
            file_hash(context.payload["basis_file"]) != context.basis_id
            or file_hash(context.payload["checkpoint"]) != context.payload["checkpoint_hash"]
            or receipt["artifacts"]["basis_file"]["sha256"] != context.basis_id
            or receipt["artifacts"]["checkpoint"]["sha256"] != context.payload["checkpoint_hash"]
        ):
            raise ContractError("PAS initial physical files differ from the signed context")
        source_case = physical_case(initial.get("source_case", context.payload["case"]))
        case = physical_case(source_case, composition="composition" in config["channels"])
        from . import electron_roles

        try:
            if context.electron_roles:
                electron_roles.verify_context(context)
            role_args = (
                dict(role_provenance=context.payload["electron_role_provenance"])
                if context.electron_roles and role_reference is None
                else dict(role_reference=electron_roles.discover_reference(seed, role_reference))
            )
        except ValueError as error:
            raise ContractError(str(error)) from error
        desired = context_from_files(
            case,
            context.payload["basis_file"],
            context.payload["checkpoint"],
            config,
            electrons=context.electrons,
            required_active=context.required_active,
            **role_args,
        )
        # These are metadata of the identical, hash-verified orbital files.
        # Re-reading a cold package must not reset transported ordering or its
        # signed final gauge; no density, response label or posterior is moved.
        desired.stable_order = context.stable_order
        if "gauge" in context.payload:
            desired.payload["gauge"] = context.payload["gauge"]
        original_metrics = receipt["metrics"]
        if "qit" in config["channels"] and "qit" not in original_metrics:
            return preceding("Cached initialization lacks basis-bound QIT")
        if context.backend != config["backend"]:
            return preceding("Cached initialization was generated by another solver backend")
        density_source = None
        if "occupation" in config["channels"]:
            if "densities" in receipt.get("artifacts", {}):
                d = np.load(verify_binding(receipt["artifacts"]["densities"]), allow_pickle=False)
                metrics.validate_densities(d, desired.states["weights"], desired.electrons)
            else:
                if context.backend == "dmrg":
                    with h5py.File(desired.payload["basis_file"]) as h:
                        native_density = "DENSITY_MATRIX" in h and np.any(h["DENSITY_MATRIX"][()])
                    if not native_density and any(
                        not Path(desired.payload["checkpoint"] + "." + str(root)).is_file()
                        for root in desired.states["roots"]
                    ):
                        return preceding(
                            "Cached DMRG initialization lacks full root densities and root natural orbitals"
                        )
                    d, _, density_source = read_dmrg_densities(desired)
                else:
                    with h5py.File(desired.payload["basis_file"]) as h:
                        if "DENSITY_MATRIX" not in h:
                            return preceding("Cached initialization lacks full per-state densities")
                    try:
                        d, _ = read_densities(
                            desired.payload["basis_file"], desired.states, desired.electrons
                        )
                    except (KeyError, FileNotFoundError):
                        return preceding("Cached initialization lacks full per-state densities")
        else:
            d = None
        values = {}
        if "occupation" in config["channels"]:
            values["occupation"] = metrics.occupation(
                d, desired.states["weights"], desired.electrons
            )
        if "composition" in config["channels"]:
            composition_metrics = metrics.evaluate(
                ["composition"],
                {"composition": 1.0},
                composition=desired.composition,
                active=desired.active,
                composition_pair=desired.composition_pair,
                composition_model=config["composition_model"],
                composition_channels=desired.composition_channels,
                composition_preferences=desired.composition_preferences,
                composition_purity_weight=config.get("composition_purity_weight", 0.5),
            )
            values.update(
                {key: value for key, value in composition_metrics.items() if key != "quality"}
            )
        evidence = receipt.get("artifacts", {}).get("mps_evidence")
        data = None
        if evidence is not None:
            data = read_json(verify_binding(evidence))
            if (
                data.get("orbital_basis_id") != desired.basis_id
                or data.get("status") != "available"
                or data.get("state_ids") != desired.states["ids"]
                or data.get("active_orbital_indices") != list(desired.active)
                or data.get("energy_source") != "expectation_of_same_final_MPS_as_density"
            ):
                raise ContractError("Cached MPS basis, state or physical energy evidence differs")
        if "qit" in config["channels"]:
            if evidence is None:
                return preceding("Cached initialization lacks bound MPS evidence")
            if (
                data.get("entropy_unit") != "natural_log"
                or data.get("mutual_information_convention") != "0.5*(s_i+s_j-s_ij)"
            ):
                raise ContractError("Cached QIT units or convention differ")
            rows = {row["state_id"]: row for row in data["states"]}
            values["qit"] = metrics.qit(
                np.asarray([rows[i]["single_orbital_entropy"] for i in desired.states["ids"]]),
                np.asarray([rows[i]["mutual_information"] for i in desired.states["ids"]]),
                desired.states["weights"],
            )
        values["quality"] = sum(config["weights"][c] * values[c] for c in config["channels"]) / sum(
            config["weights"].values()
        )
        if (
            config["backend"] == "cas"
            and config.get("survey_model") == "posterior_evidence_capture_v1"
        ):
            values["qit"] = None
        if data is None:
            values["energy"] = _energy_check(
                desired.payload["basis_file"], desired.states, original_metrics.get("energy")
            )
        else:
            _energy_check(desired.payload["basis_file"], desired.states)
            energies = np.asarray(data.get("root_energies", []), float)
            if energies.shape != (len(desired.states["ids"]),) or not np.isfinite(energies).all():
                raise ContractError("Cached MPS energies are missing or nonfinite")
            values["energy"] = float(energies @ desired.states["weights"])
            if not np.isclose(values["energy"], original_metrics.get("energy"), atol=1e-7, rtol=0):
                raise ContractError(
                    "Cached terminal MPS energy differs from initialization receipt"
                )
        # Identical replay is identity-preserving. A new mask/protocol creates a
        # fresh initial evidence receipt from files, never a migrated posterior.
        reused_new_cost = Path(output_directory).resolve() != source.parent and (
            initial.get("cpu_seconds", 0) != 0 or initial.get("wall_seconds", 0) != 0
        )
        if (
            desired.context_id != context.context_id
            or values != original_metrics
            or reused_new_cost
            or density_source is not None
        ):
            receipt = copy.deepcopy(receipt)
            initial = copy.deepcopy(initial)
            receipt.update(
                kind="initial",
                context_id=desired.context_id,
                output_context=desired.to_dict(),
                metrics=values,
            )
            if density_source is not None:
                output = Path(output_directory).resolve()
                output.mkdir(parents=True, exist_ok=True)
                np.save(output / "densities.npy", d)
                receipt["artifacts"]["densities"] = binding(output / "densities.npy")
                for role, item in density_source["sources"].items():
                    receipt["artifacts"]["density_" + role] = item
                initial["density_reconstruction"] = density_source
            initial["generation_protocol"] = initial.get(
                "generation_protocol", initial.get("protocol")
            )
            initial["matches_search_protocol"] = bool(
                initial.get("matches_search_protocol")
                and desired.protocol_id == context.protocol_id
            )
            initial["source_seed"] = binding(source)
            initial["source_case"] = source_case
            if reused_new_cost:
                for cost in ("cpu_seconds", "wall_seconds"):
                    prior = initial.get("historical_" + cost)
                    unknown = initial.get("historical_costs_unknown", False)
                    initial["known_historical_" + cost] = initial.get(cost, 0) + (prior or 0)
                    initial["historical_" + cost] = (
                        None if unknown else initial["known_historical_" + cost]
                    )
                    initial[cost] = 0.0
                    receipt[cost] = 0.0
                initial["role"] = "reused_frozen_cold_stage"
            receipt["provenance"] = initial
            receipt["evidence_id"] = identity(
                {k: v for k, v in receipt.items() if k != "evidence_id"}
            )
            seed = dict(context=desired.to_dict(), incumbent=receipt, initialization=initial)
        output = Path(output_directory).resolve()
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "seed.json", seed)
        return seed
    s = seed["snapshot"]
    e = seed["incumbent"]
    if not all(e.get(k) for k in ("valid", "converged", "state_valid")):
        return preceding("Historical initialization lacks a valid convergence/state receipt")
    historical_protocol = seed.get("initialization_protocol")
    if (
        historical_protocol == "fresh_scf_then_M1000_sa2_dmrgscf"
        or s["case"].get("solver", {}).get("ci_solver") == "block2"
    ):
        if (
            historical_protocol != "fresh_scf_then_M1000_sa2_dmrgscf"
            or seed.get("reused_search_results") is not False
            or seed.get("initialization_cost", {}).get("initial_evidence_id")
            != e.get("evidence_id")
        ):
            raise ContractError("Historical DMRG initialization identity is ambiguous")
        # This marker identifies a generation route, not a complete physical
        # receipt. Its old request/log/endpoint assets have not been migrated
        # into our signed initial-stage contract; do not relabel it as CAS/QN.
        reason = (
            "Historical DMRG initial selection used composition or has unknown dependencies"
            if "composition" not in config["channels"]
            else "Historical DMRG request, log and endpoint bindings are not yet imported"
        )
        return preceding(reason)
    if config["backend"] != "cas":
        return preceding("Historical CASSCF initialization was generated by another solver backend")
    try:
        _verify_seed_provenance(seed)
    except (FileNotFoundError, InitializationUnavailable):
        return preceding("Historical initialization is missing its required bound evidence")
    selection = s["case"].get("cold_start", {})
    if "composition" not in config["channels"] and selection.get("target_weight", None) != 0:
        return preceding(
            "Historical initial selection used composition or has unknown dependencies"
        )
    for field, digest in [("basis_file", "basis_id"), ("checkpoint", "checkpoint_hash")]:
        if file_hash(s[field]) != s[digest]:
            raise ContractError("Imported physical artifact changed")
    if s["budget"] != [16, 13] or s["backend"] != "openmolcas":
        raise ContractError("Four-point import requires OpenMolcas CAS(16e,13o)")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    basis = _copy_artifact(s["basis_file"], output / "initial.rasscf.h5")
    checkpoint = _copy_artifact(s["checkpoint"], output / "initial.RasOrb")
    from . import electron_roles

    try:
        role_reference = electron_roles.discover_reference(seed, role_reference)
    except ValueError as error:
        raise ContractError(str(error)) from error
    context = context_from_files(
        physical_case(
            s["case"],
            composition="composition" in config["channels"],
            active_space=dict(electrons=16, orbitals=13),
        ),
        basis,
        checkpoint,
        config,
        electrons=16,
        role_reference=role_reference,
    )
    if len(context.active) != 13 or list(context.active) != s["active"]:
        raise ContractError("Imported active partition differs")
    energy = _energy_check(basis, context.states, e["metrics"]["energy"])
    if "qit" in config["channels"]:
        return preceding("Historical exact-CAS seed has no basis-bound QIT")
    densities = None
    if "occupation" in config["channels"]:
        with h5py.File(basis) as h:
            if "DENSITY_MATRIX" not in h:
                return preceding("Historical initialization lacks full per-state densities")
        densities, _ = read_densities(basis, context.states, 16)
        np.save(output / "densities.npy", densities)
    values = metrics.evaluate(
        config["channels"],
        config["weights"],
        densities=densities,
        state_weights=context.states["weights"],
        electrons=16,
        composition=context.composition,
        active=context.active,
        composition_pair=context.composition_pair,
        composition_channels=context.composition_channels,
        composition_model=config["composition_model"],
        composition_preferences=context.composition_preferences,
        composition_purity_weight=config.get("composition_purity_weight", 0.5),
    )
    values["energy"] = energy
    if config.get("survey_model") == "posterior_evidence_capture_v1":
        values["qit"] = None
    initial = dict(
        source_seed=binding(source),
        role="imported_verified_cold_stage",
        fresh_scf=False,
        source_case=physical_case(s["case"], active_space=dict(electrons=16, orbitals=13)),
        solver_executions=0,
        historical_costs_unknown=seed.get("historical_initialization_cpu_seconds") is None,
        historical_cpu_seconds=seed.get("historical_initialization_cpu_seconds"),
        historical_wall_seconds=seed.get("historical_initialization_wall_seconds"),
        cpu_seconds=0.0,
        wall_seconds=0.0,
        source_protocol_not_current_reverified=True,
        reported_source_solver_settings=s["case"].get("solver"),
        historical_protocol_verified=False,
        historical_optimizer="not_relabelled_or_inferred",
        initial_selection_used_composition=selection.get("target_weight", None) != 0,
        matches_search_protocol=False,
        electron_role_reference=role_reference,
    )
    receipt = dict(
        request_id=identity(initial),
        context_id=context.context_id,
        kind="initial",
        active=list(context.active),
        valid=True,
        converged=True,
        state_valid=True,
        metrics=values,
        cpu_seconds=0.0,
        wall_seconds=0.0,
        output_context=context.to_dict(),
        artifacts=dict(basis_file=binding(basis), checkpoint=binding(checkpoint)),
        provenance=initial,
    )
    if densities is not None:
        receipt["artifacts"]["densities"] = binding(output / "densities.npy")
    receipt["evidence_id"] = identity(receipt)
    result = dict(context=context.to_dict(), incumbent=receipt, initialization=initial)
    write_json(output / "seed.json", result)
    return result


_CAS_SURVEY_INPUT = "Passive analysis of the parent CAS wavefunction 1RDM; no solver input or orbital optimization.\n"


def _cas_survey_source(context, partition):
    return dict(
        schema="pas_parent_cas_density_analysis_v1",
        passive=True,
        evidence_source="parent_CAS_wavefunction",
        correlation_scope="parent_active",
        correlated_orbitals=list(context.active),
        parent_electrons=context.electrons,
        source_basis=binding(context.payload["basis_file"]),
        source_checkpoint=binding(context.payload["checkpoint"]),
        analyzed_partition=partition,
        density_construction="parent_active_1RDM_plus_inactive_2_secondary_0",
        density_gauge="verified_native_SA_natural_orbitals",
        qit=None,
        orbital_optimization=False,
        physical_jobs_launched=0,
        energy_used_for_reward=False,
    )


def _cas_survey_density(context, partition):
    """Restrict the existing CAS wavefunction's full 1RDM, without solving a larger CAS."""
    measured = partition["active"]
    if not set(context.active) <= set(measured):
        raise ContractError(
            "Parent CAS density analysis must retain the complete correlated active block"
        )
    parent, energies = read_densities(
        context.payload["basis_file"], context.states, context.electrons
    )
    if parent.shape != (len(context.states["roots"]), context.k, context.k):
        raise ContractError("Parent CAS density does not match the physical active partition")
    lookup = {orbital: i for i, orbital in enumerate(measured)}
    active = [lookup[i] for i in context.active]
    density = np.zeros((len(parent), len(measured), len(measured)), dtype=parent.dtype)
    density[:, np.asarray(active)[:, None], active] = parent
    for orbital in set(context.inactive) & set(measured):
        density[:, lookup[orbital], lookup[orbital]] = 2.0
    metrics.validate_densities(density, context.states["weights"], partition["electrons"])
    return density, energies


def verify_cas_survey(context, partition, config, request, receipt):
    """Verify passive provenance and its derived statistics against the immutable parent."""
    if (
        context.backend != "cas"
        or config.get("survey_model") != "posterior_evidence_capture_v1"
        or request.get("input") != _CAS_SURVEY_INPUT
        or request.get("analysis") != _cas_survey_source(context, partition)
        or receipt.get("provenance") != request["analysis"]
        or receipt.get("target_response_label") is not False
        or receipt.get("physical_jobs_launched") != 0
        or receipt.get("metrics", {}).get("qit") is not None
        or "output_context" in receipt
        or not all(receipt.get(k) for k in ("valid", "converged", "state_valid"))
    ):
        raise ContractError("CAS survey is not the declared passive parent-density analysis")
    artifacts = receipt.get("artifacts", {})
    for key, source in (("basis_file", "source_basis"), ("checkpoint", "source_checkpoint")):
        if artifacts.get(key) != request["analysis"][source]:
            raise ContractError("CAS survey source artifact differs from its parent")
        verify_binding(artifacts[key])
    density, energies = _cas_survey_density(context, partition)
    saved = np.load(verify_binding(artifacts["densities"]), allow_pickle=False)
    if not np.array_equal(saved, density):
        raise ContractError("CAS survey density differs from the parent wavefunction embedding")
    expected = survey_quadratic(context, partition, density, None, None, config)
    if receipt.get("proposal_quadratic") != expected:
        raise ContractError("CAS survey posterior differs from its parent density")
    values = metrics.evaluate(
        config["channels"],
        config["weights"],
        densities=density,
        state_weights=context.states["weights"],
        electrons=partition["electrons"],
        composition=context.composition,
        active=partition["active"],
        composition_pair=context.composition_pair,
        composition_channels=context.composition_channels,
        composition_model=config["composition_model"],
        composition_preferences=context.composition_preferences,
        composition_purity_weight=config.get("composition_purity_weight", 0.5),
    )
    values.update(energy=float(energies @ context.states["weights"]), qit=None)
    if receipt["metrics"] != values:
        raise ContractError("CAS survey diagnostic metrics differ from its parent density")
    timing = read_json(verify_binding(artifacts["execution"]))
    if timing.get("physical_jobs_launched") != 0 or timing.get("return_code") != 0:
        raise ContractError("CAS survey cannot claim a physical solver execution")
    for key in ("cpu_seconds", "wall_seconds"):
        if receipt.get(key) != timing.get(key):
            raise ContractError("CAS survey analysis cost differs from its execution record")


def _execute_cas_survey(context, partition, directory, config, request):
    stored = directory / "passive_result.json"
    if (directory / "execution.json").exists():
        if not stored.is_file():
            raise RuntimeError("Incomplete passive CAS analysis; inspect before explicit recovery")
        result = read_json(stored)
        if result.pop("result_id") != identity(result):
            raise ContractError("Passive CAS analysis content changed")
    else:
        if _RECONCILE_ONLY.get():
            raise RuntimeError("Passive CAS analysis lacks a completed result")
        start, cpu = time.perf_counter(), time.process_time()
        density, energies = _cas_survey_density(context, partition)
        values = metrics.evaluate(
            config["channels"],
            config["weights"],
            densities=density,
            state_weights=context.states["weights"],
            electrons=partition["electrons"],
            composition=context.composition,
            active=partition["active"],
            composition_pair=context.composition_pair,
            composition_channels=context.composition_channels,
            composition_model=config["composition_model"],
            composition_preferences=context.composition_preferences,
            composition_purity_weight=config.get("composition_purity_weight", 0.5),
        )
        values.update(energy=float(energies @ context.states["weights"]), qit=None)
        fields = survey_quadratic(context, partition, density, None, None, config)
        np.save(directory / "densities.npy", density)
        timing = dict(
            cpu_seconds=time.process_time() - cpu,
            wall_seconds=time.perf_counter() - start,
            return_code=0,
            physical_jobs_launched=0,
        )
        write_json(directory / "execution.json", timing)
        result = dict(
            request_id=request["request_id"],
            context_id=context.context_id,
            kind="survey",
            active=partition["active"],
            valid=True,
            converged=True,
            state_valid=True,
            metrics=values,
            proposal_quadratic=fields,
            target_response_label=False,
            **timing,
            provenance=request["analysis"],
            artifacts=dict(
                request=binding(directory / "request.json"),
                basis_file=request["analysis"]["source_basis"],
                checkpoint=request["analysis"]["source_checkpoint"],
                densities=binding(directory / "densities.npy"),
                execution=binding(directory / "execution.json"),
            ),
        )
        write_json(stored, dict(result, result_id=identity(result)))
    if result.get("request_id") != request["request_id"]:
        raise ContractError("Passive CAS result differs from its request")
    verify_cas_survey(context, partition, config, request, result)
    result["artifacts"]["passive_result"] = binding(stored)
    return result


def _execute_cas(context, partition, kind, directory, config, request):
    """Execute or reconcile one CAS action and construct its unsigned receipt."""
    timing = (
        read_json(directory / "execution.json")
        if (directory / "execution.json").exists()
        else run_solver(directory, request["input"], config["threads"])
    )
    log = (directory / "solver.log").read_text(errors="replace")
    fixed = kind != "optimize" or config.get("orbital_optimization") is False
    converged = _converged(log, "probe" if fixed else kind, timing["return_code"])
    receipt = dict(
        request_id=request["request_id"],
        context_id=context.context_id,
        kind=kind,
        active=partition["active"],
        valid=False,
        converged=converged,
        state_valid=False,
        metrics={},
        **timing,
        artifacts=dict(
            request=binding(directory / "request.json"),
            log=binding(directory / "solver.log"),
            execution=binding(directory / "execution.json"),
        ),
    )
    if not converged:
        receipt["failure"] = "native_numerical_nonconvergence"
    else:
        basis = directory / "job.rasscf.h5"
        checkpoint = directory / "job.RasOrb"
        output = context_from_files(
            context.payload["case"],
            basis,
            checkpoint,
            config,
            electrons=partition["electrons"],
            reference_context=context,
            input_partition=partition,
        )
        if len(output.active) != context.k:
            raise ContractError("Solver changed active-space size")
        mapping = _audit_output(context, output, partition, "probe" if fixed else kind)
        if fixed:
            output.payload["gauge"] = "fixed_input_MO_basis_v1"
        qn = verify_qn(log) if not fixed else None
        d = None
        if "occupation" in config["channels"]:
            d, energies = read_densities(
                basis, context.states, partition["electrons"], require_natural=not fixed
            )
            np.save(directory / "densities.npy", d)
        else:
            _energy_check(basis, context.states)
            with h5py.File(basis) as h:
                energies = np.asarray(h["ROOT_ENERGIES"][()], float).reshape(-1)
        values = metrics.evaluate(
            config["channels"],
            config["weights"],
            densities=d,
            state_weights=context.states["weights"],
            electrons=partition["electrons"],
            composition=output.composition,
            active=output.active,
            composition_pair=output.composition_pair,
            composition_channels=output.composition_channels,
            composition_model=config["composition_model"],
            composition_preferences=output.composition_preferences,
            composition_purity_weight=config.get("composition_purity_weight", 0.5),
        )
        values["energy"] = float(energies @ context.states["weights"])
        if config.get("survey_model") == "posterior_evidence_capture_v1":
            values["qit"] = None
        receipt.update(
            valid=True, state_valid=True, metrics=values, output_context=output.to_dict()
        )
        receipt["artifacts"].update(basis_file=binding(basis), checkpoint=binding(checkpoint))
        if d is not None:
            receipt["artifacts"]["densities"] = binding(directory / "densities.npy")
        receipt["provenance"] = dict(
            density_gauge=(
                "verified_fixed_input_MO_basis" if fixed else "verified_native_SA_natural_orbitals"
            )
            if d is not None
            else None,
            input_partition=partition,
            output_active=list(output.active),
            energy_used_for_reward=False,
            basis_mapping=mapping,
            qn_verified=bool(qn),
            qn=qn,
        )
        if config.get("orbital_optimization") is False:
            receipt["provenance"].update(orbital_optimization=False, method="CASCI")
    return receipt


def execute(context, active, kind, directory, config, *, fixed_dmrg_policy=None):
    config = configuration(config)
    from .preferences import assert_preference_binding

    assert_preference_binding(context, config)
    fixed_options = {}
    if fixed_dmrg_policy is not None:
        from .solvers.block2_convergence import validate_fixed_policy

        fixed_dmrg_policy = validate_fixed_policy(fixed_dmrg_policy)
        if config["backend"] != "dmrg":
            raise ContractError("Fixed DMRG policy requires the DMRG backend")
        fixed_options = dict(fixed_dmrg_policy=fixed_dmrg_policy)
    if not isinstance(context, Context):
        context = Context.from_dict(context)
    if config["backend"] == "pyscf":
        from .pyscf_backend import execute as pyscf_execute

        return pyscf_execute(context, active, kind, directory, config)
    if kind not in ("probe", "optimize", "survey"):
        raise ContractError("Unknown physical action")
    passive_cas = (
        kind == "survey"
        and context.backend == "cas"
        and config.get("survey_model") == "posterior_evidence_capture_v1"
    )
    if kind == "survey" and context.backend != "dmrg" and not passive_cas:
        raise ContractError("Native exact CAS survey is unsupported")
    check_capabilities(context, kind, config["channels"], config)
    _check_context(context, config)
    if kind == "survey":
        from .diagnostic_window import survey_partition

        partition = survey_partition(context, active)
    else:
        partition = context.partition(active)
    directory = checked_directory(directory, config["threads"])
    request = dict(
        schema="pas_physical_request_v1",
        context_id=context.context_id,
        basis_id=context.basis_id,
        checkpoint_hash=context.payload["checkpoint_hash"],
        kind=kind,
        partition=partition,
        protocol=config,
        input=_CAS_SURVEY_INPUT if passive_cas else input_text(context, partition, kind, config),
    )
    if passive_cas:
        request["analysis"] = _cas_survey_source(context, partition)
    request.update(fixed_options)
    request["request_id"] = identity(request)
    if (directory / "receipt.json").exists():
        receipt = read_json(directory / "receipt.json")
        if receipt["request_id"] != request["request_id"]:
            raise ContractError("Cached physical request differs")
        if receipt.get("fixed_dmrg_policy") != fixed_dmrg_policy:
            raise ContractError("Cached fixed DMRG policy differs")
        if receipt.get("evidence_id") != identity(
            {k: v for k, v in receipt.items() if k != "evidence_id"}
        ):
            raise ContractError("Cached physical receipt content changed")
        for item in receipt.get("artifacts", {}).values():
            if isinstance(item, dict) and {"path", "sha256"} <= item.keys():
                verify_binding(item)
        if passive_cas:
            verify_cas_survey(context, partition, config, request, receipt)
        return receipt
    if (directory / "request.json").exists():
        if read_json(directory / "request.json") != request:
            raise ContractError("Existing request differs")
        if not (directory / "execution.json").exists():
            raise RuntimeError("Unreconciled execution cannot be rerun")
        if not _RECONCILE_ONLY.get():
            return reconcile(context, active, kind, directory, config, **fixed_options)
    else:
        if _RECONCILE_ONLY.get():
            raise RuntimeError("Unresolved physical request: missing frozen request record")
        write_json(directory / "request.json", request)
        if not passive_cas:
            actual = write_partition(
                context, active, directory / "trial.RasOrb", survey=kind == "survey"
            )
            if actual != partition:
                raise ContractError("Input orbital partition differs from frozen request")
            write_bound_geometry(context, directory / "geometry.xyz")
    if passive_cas:
        receipt = _execute_cas_survey(context, partition, directory, config, request)
    elif context.backend == "dmrg":
        receipt = _execute_dmrg(
            context, partition, kind, directory, config, request, **fixed_options
        )
    else:
        receipt = _execute_cas(context, partition, kind, directory, config, request)
    receipt.update(fixed_options)
    receipt["evidence_id"] = identity(receipt)
    write_json(directory / "receipt.json", receipt)
    return receipt


def reconcile(context, active, kind, directory, config, *, fixed_dmrg_policy=None):
    """Consume complete recorded outputs only; never dispatch a missing stage."""
    directory = Path(directory).resolve()
    if not (directory / "request.json").is_file():
        raise RuntimeError("Unresolved pending request: no physical request record")
    if (directory / "started.json").exists():
        started = read_json(directory / "started.json")
        pid = started.get("solver_pid")
        if pid and Path(f"/proc/{pid}/stat").exists():
            stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if stat[19] == started.get("solver_start_ticks") and stat[0] != "Z":
                raise RuntimeError("Unresolved pending request: its physical solver is still live")
    if not (directory / "execution.json").is_file():
        raise RuntimeError("Unresolved pending request: no completed execution record")
    token = _RECONCILE_ONLY.set(True)
    options = {} if fixed_dmrg_policy is None else dict(fixed_dmrg_policy=fixed_dmrg_policy)
    try:
        return execute(context, active, kind, directory, config, **options)
    finally:
        _RECONCILE_ONLY.reset(token)


def fcidump_header(path):
    """Read only the header; surveying a wide window must not parse all ERIs twice."""
    lines = []
    with Path(path).open() as stream:
        for line in stream:
            lines.append(line)
            if "/" in line or re.search(r"&END", line, re.I):
                break
        else:
            raise ContractError("Unterminated FCIDUMP header")
    header = " ".join(lines)
    return {
        key: int(re.search(r"\b" + key + r"\s*=\s*(-?\d+)", header, re.I)[1])
        for key in ("NORB", "NELEC", "MS2")
    }


def fcidump(path):
    text = Path(path).read_text()
    end = text.find("/")
    if end < 0:
        match = re.search(r"&END", text, re.I)
        if match is None:
            raise ContractError("Unterminated FCIDUMP header")
        end = match.end() - 1
    header = {
        key: int(re.search(r"\b" + key + r"\s*=\s*(-?\d+)", text[: end + 1], re.I)[1])
        for key in ("NORB", "NELEC", "MS2")
    }
    records = {}
    for line in text[end + 1 :].splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ContractError("Malformed FCIDUMP integral")
        key = tuple(map(int, fields[1:]))
        value = float(fields[0].replace("D", "E").replace("d", "e"))
        if not np.isfinite(value) or (key in records and abs(records[key] - value) > 1e-13):
            raise ContractError("Nonfinite/conflicting FCIDUMP integral")
        records[key] = value
    if not records:
        raise ContractError("Empty FCIDUMP")
    return header, records


def _dmrg_environment():
    return dict(
        MOLCAS=MOLCAS,
        PATH=os.environ["PATH"] + os.pathsep + str(Path(BLOCK2MAIN).parent),
        PYTHONPATH=str(Path(__file__).resolve().parents[1])
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
    )


def verify_export_input_basis(source_native, input_types, actual, tags):
    """Verify decoded native input and its deterministic within-active order."""
    order = [i for tag in "fi123sd" for i, t in enumerate(input_types) if t == tag]
    if (
        len(order) != source_native.shape[1]
        or list(tags) != list(input_types)
        or actual.shape != source_native.shape
        or not np.allclose(actual, source_native, atol=1e-10, rtol=0)
    ):
        raise ContractError(
            "Native integral input MO basis or active ordering differs from requested input"
        )
    return order


def _export_integrals(
    context,
    partition,
    source_checkpoint,
    directory,
    config,
    *,
    dimension=None,
    fixed_dmrg_policy=None,
):
    """Capture actual integrals before the external solver; exit is not CI success."""
    options = {}
    if fixed_dmrg_policy is not None:
        from .solvers.block2_convergence import validate_fixed_policy

        options["fixed_dmrg_policy"] = validate_fixed_policy(fixed_dmrg_policy)
        if type(dimension) is not int or dimension <= 0:
            raise ContractError("Native fixed export requires its bond dimension")
    if dimension is not None:
        if type(dimension) is not int or dimension <= 0:
            raise ContractError("Invalid integral export bond dimension")
        options["dimension"] = dimension
    source_mo = read_openmolcas_orbital_hdf5(context.payload["basis_file"])
    source_native, _ = read_rasorb_coefficients(
        source_checkpoint, symmetry_block_sizes=source_mo.symmetry_block_sizes
    )
    if not np.allclose(
        source_mo.native_coefficients_to_ao(source_native),
        source_mo.mo_coefficients,
        atol=1e-8,
        rtol=0,
    ):
        raise ContractError(
            "Integral export checkpoint coefficients differ from declared input basis"
        )
    fixed_record_proof = verify_native_inporb22(source_checkpoint, source_native)
    directory = Path(directory).resolve()
    cached = directory / "integrals.json"
    if cached.exists():
        receipt = read_json(cached)
        if receipt.get("export_id") != identity(
            {k: v for k, v in receipt.items() if k != "export_id"}
        ):
            raise ContractError("Integral export receipt content changed")
        if (
            receipt["basis_id"] != context.basis_id
            or receipt["source_checkpoint"] != binding(source_checkpoint)
            or receipt["partition"] != partition
            or receipt.get("config_id") != identity(config)
            or receipt.get("dimension") != dimension
            or receipt.get("fixed_dmrg_policy") != fixed_dmrg_policy
        ):
            raise ContractError("Integral export input identity changed")
        for name in (
            "integrals",
            "native_input_orbitals",
            "native_hdf5",
            "native_parser_executable",
        ):
            verify_binding(receipt[name])
        if fixed_dmrg_policy is not None:
            verify_binding(receipt["native_configuration"])
            verify_binding(receipt["request"])
        return receipt
    _allow_dispatch()
    directory = checked_directory(directory, config["threads"])
    shutil.copy2(source_checkpoint, directory / "trial.RasOrb")
    write_bound_geometry(context, directory / "geometry.xyz")
    wrapper = directory / "capture_bin"
    wrapper.mkdir(exist_ok=True)
    script = wrapper / "block2main"
    script.write_text(
        "#!" + sys.executable + "\nfrom pas.solvers.integral_capture import main\nmain()\n"
    )
    script.chmod(0o700)
    env = _dmrg_environment()
    env.update(PATH=str(wrapper) + os.pathsep + env["PATH"], PAS_INTEGRAL_CAPTURE=str(directory))
    export_config = config if dimension is None else {**config, "dmrg_probe_m": dimension}
    text = input_text(context, partition, "probe", export_config)
    if options:
        request = dict(
            basis_id=context.basis_id,
            partition=partition,
            source_checkpoint=binding(source_checkpoint),
            config_id=identity(config),
            input=text,
            **options,
        )
        path = directory / "export_request.json"
        if path.exists() and read_json(path) != request:
            raise ContractError("Integral export request changed")
        if not path.exists() and (directory / "execution.json").exists():
            raise ContractError("Native integral export lacks its frozen dimension/policy request")
        write_json(path, request)
    if (directory / "execution.json").exists():
        # Completed integral-only work may be extracted without another native
        # invocation. An incomplete capture is an error, never a rerun trigger.
        timing = read_json(directory / "execution.json")
    else:
        timing = run_solver(directory, text, config["threads"], environment=env)
    if not (directory / "integral_capture.json").is_file():
        raise ContractError(
            "Native integral capture did not complete; inspect " + str(directory / "solver.log")
        )
    capture = read_json(directory / "integral_capture.json")
    dump = directory / "FCIDUMP"
    if capture["dmrg_sweeps_executed"] != 0 or capture["fcidump_sha256"] != file_hash(dump):
        raise ContractError("Integral capture has invalid provenance")
    if fixed_dmrg_policy is not None:
        native_configuration = directory / "export_dmrg.conf"
        if not native_configuration.is_file() or capture.get("configuration_sha256") != file_hash(
            native_configuration
        ):
            raise ContractError("Native integral configuration capture binding differs")
    actual_orbitals = directory / "native_input.InpOrb"
    if capture.get("native_input_orbitals_sha256") != file_hash(actual_orbitals):
        raise ContractError("Native consumed orbital input changed")
    if file_hash(actual_orbitals) != file_hash(source_checkpoint):
        raise ContractError("Consumed native INPORB bytes differ from the requested checkpoint")
    actual, tags = read_rasorb_coefficients(
        actual_orbitals, symmetry_block_sizes=source_mo.symmetry_block_sizes
    )
    input_types = read_inporb(source_checkpoint).type_indices
    verify_native_inporb22(actual_orbitals, source_native)
    order = verify_export_input_basis(source_native, input_types, actual, tags)
    expected_tags = [""] * context.nmo
    for key, tag in [
        ("active", "2"),
        ("inactive", "i"),
        ("frozen", "f"),
        ("secondary", "s"),
        ("deleted", "d"),
    ]:
        for i in partition[key]:
            expected_tags[i] = tag
    if list(input_types) != expected_tags:
        raise ContractError("Native integral input partition differs from requested scope")
    orthogonal_error = float(
        abs(
            source_mo.mo_coefficients.T @ source_mo.ao_overlap @ source_mo.mo_coefficients
            - np.eye(context.nmo)
        ).max()
    )
    if orthogonal_error > 1e-10:
        raise ContractError("Integral input basis is not AO-metric orthonormal")
    native_hdf5 = directory / "work/job/job.rasscf.h5"
    actual_meta = read_openmolcas_orbital_hdf5(native_hdf5)
    _geometry(context.payload["case"], actual_meta, native_hdf5)
    if not np.array_equal(
        actual_meta.basis_function_ids, source_mo.basis_function_ids
    ) or not np.allclose(actual_meta.ao_overlap, source_mo.ao_overlap, atol=1e-10, rtol=0):
        raise ContractError("Native integral export AO basis/metric differs")
    header = fcidump_header(dump)
    if header != dict(
        NORB=len(partition["active"]),
        NELEC=partition["electrons"],
        MS2=context.states["spin_multiplicity"] - 1,
    ):
        raise ContractError("Exported scope electron/orbital/spin numbers differ")
    receipt = dict(
        schema="pas_integral_export_v1",
        basis_id=context.basis_id,
        source_checkpoint=binding(source_checkpoint),
        partition=partition,
        integrals=binding(dump),
        geometry_id=context.geometry_id,
        states=context.states,
        orbital_order="pinned_native_stable_type_grouping_restricted_to_active",
        basis_proof_method="input_basis_semantics_verified_not_native_MO_readback",
        native_fixed_record_proof=fixed_record_proof,
        native_input_orbitals=binding(actual_orbitals),
        native_hdf5=binding(native_hdf5),
        native_type_grouping_order=order,
        config_id=identity(config),
        native_coefficient_max_error=float(abs(actual - source_native).max()),
        native_coefficient_absolute_tolerance=1e-10,
        native_input_options=["LumOrb", "TypeIndex", "CIONLY"],
        input_AO_orthogonality_max_error=orthogonal_error,
        input_AO_orthogonality_absolute_tolerance=1e-10,
        native_orthogonalization_semantics="pinned_ONCMO_Gram_Schmidt_identity_for_orthonormal_input_up_to_roundoff",
        native_parser_executable=binding(Path(MOLCAS) / "bin/rasscf.exe"),
        dmrg_sweeps_executed=0,
        **timing,
    )
    receipt.update(options)
    if fixed_dmrg_policy is not None:
        receipt["native_configuration"] = binding(native_configuration)
        receipt["request"] = binding(directory / "export_request.json")
    receipt["export_id"] = identity(receipt)
    write_json(directory / "integrals.json", receipt)
    return receipt


def _extract_mps(work, context, partition, directory, config):
    _allow_dispatch()
    import resource

    from .solvers.block2_checkpoint import REQUEST_SCHEMA

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    qit = "qit" in config["channels"]
    request = dict(
        schema=REQUEST_SCHEMA,
        source_directory=str(Path(work).resolve()),
        orbital_basis_id=context.basis_id,
        physical_evidence_id=context.context_id,
        active_orbital_indices=partition["active"],
        active_electrons=partition["electrons"],
        spin=context.states["spin_multiplicity"] - 1,
        state_ids=context.states["ids"],
        state_weights=context.states["weights"],
        checkpoint=dict(
            symmetry="SU2",
            mps_tag="KET",
            threads=config["threads"],
            stack_memory_bytes=4_000_000_000,
        ),
        requested_evidence=["single_orbital_entropy", "mutual_information"]
        if qit
        else ["occupation"],
        extraction_strategies=["checkpoint_mps"],
    )
    request["request_id"] = identity(request)
    write_json(directory / "request.json", request)
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    with (directory / "extract.log").open("w") as log:
        job = subprocess.run(
            [
                sys.executable,
                "-X",
                "faulthandler",
                "-u",
                "-m",
                "pas.solvers.block2_checkpoint",
                "--request",
                str(directory / "request.json"),
                "--result",
                str(directory / "result.json"),
                "--temporary-directory",
                str(directory / "scratch"),
            ],
            env={
                **os.environ,
                **_dmrg_environment(),
                **{
                    key: str(config["threads"])
                    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
                },
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    write_json(
        directory / "execution.json",
        dict(
            return_code=job.returncode,
            cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
            wall_seconds=time.perf_counter() - start,
        ),
    )
    if job.returncode:
        raise ContractError(
            f"MPS evidence extraction failed (return code {job.returncode}); no proxy substituted: "
            + str(directory / "extract.log")
        )
    result = read_json(directory / "result.json")
    if (
        result["request_id"] != request["request_id"]
        or result["status"] != "available"
        or not result["source_checkpoint_unchanged"]
    ):
        raise ContractError("MPS evidence request/identity mismatch")
    if (
        result.get("orbital_basis_id") != context.basis_id
        or result.get("physical_evidence_id") != context.context_id
        or result.get("active_orbital_indices") != partition["active"]
        or result.get("state_ids") != context.states["ids"]
        or result.get("state_weights") != context.states["weights"]
        or result.get("result_id")
        != identity({k: v for k, v in result.items() if k != "result_id"})
    ):
        raise ContractError("MPS evidence basis/state/content binding mismatch")
    d = {
        order["bra_root"]: matrix
        for order, matrix in zip(
            result["state_operator_order"], result["state_operators"], strict=True
        )
        if order["bra_root"] == order["ket_root"]
    }
    if set(d) != set(range(len(context.states["roots"]))):
        raise ContractError("MPS lacks complete per-root densities")
    densities = np.asarray([d[i] for i in sorted(d)])
    metrics.validate_densities(densities, context.states["weights"], partition["electrons"])
    entropies = mi = None
    if qit:
        if (
            result.get("entropy_unit") != "natural_log"
            or result.get("mutual_information_convention") != "0.5*(s_i+s_j-s_ij)"
        ):
            raise ContractError("Unsupported QIT units/convention")
        rows = {r["state_id"]: r for r in result["states"]}
        entropies = np.asarray([rows[s]["single_orbital_entropy"] for s in context.states["ids"]])
        mi = np.asarray([rows[s]["mutual_information"] for s in context.states["ids"]])
        metrics.qit(entropies, mi, context.states["weights"])
    return dict(
        densities=densities,
        entropies=entropies,
        mutual_information=mi,
        result=result,
        cpu_seconds=after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime,
        wall_seconds=time.perf_counter() - start,
        artifact=binding(directory / "result.json"),
    )


def _check_native_fixed_gate(gate, policy, native_config, nroots, return_code):
    """Authenticate the all-root, center-zero endpoint for the native protocol."""
    if (
        not isinstance(gate, dict)
        or gate.get("fixed_dmrg_policy") != policy
        or gate.get("native_config") != native_config
        or not gate.get("stages")
    ):
        raise ContractError("Native fixed endpoint protocol binding differs")
    stages = gate["stages"]
    tolerance = float(native_config["sweep_tol"])
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ContractError("Invalid native fixed tolerance")
    for stage in stages:
        deltas = np.asarray(stage.get("all_root_deltas", []), float)
        if (
            deltas.shape != (nroots,)
            or not np.all(np.isfinite(deltas))
            or np.any(deltas < 0)
            or stage.get("native_tolerance") != tolerance
            or stage.get("target_center") != 0
            or type(stage.get("sweeps")) is not int
            or type(stage.get("stage_sweep_limit")) is not int
            or not 2 <= stage["sweeps"] <= stage["stage_sweep_limit"] <= native_config["maxiter"]
        ):
            raise ContractError("Malformed native fixed all-root endpoint")
        if stage.get("passed") and (stage.get("final_center") != 0 or max(deltas) >= tolerance):
            raise ContractError("Native fixed endpoint did not satisfy all roots at center zero")
    if return_code == 96:
        if (
            gate.get("passed") is not False
            or any(not s.get("passed") for s in stages[:-1])
            or stages[-1].get("passed") is not False
            or not stages[-1].get("failure", "").startswith("NumericalNonconvergence:")
            or stages[-1]["sweeps"] != stages[-1]["stage_sweep_limit"]
            or (
                stages[-1].get("final_center") == 0
                and max(stages[-1]["all_root_deltas"]) < tolerance
            )
        ):
            raise ContractError("Native fixed numerical failure has no matching endpoint proof")
        return False
    if return_code != 0:
        raise RuntimeError("Block2 process failed outside numerical convergence")
    if gate.get("passed") is not True or not all(s.get("passed") is True for s in stages):
        raise ContractError("Native fixed success lacks a completed endpoint proof")
    return True


def _fixed_dmrg(
    context, partition, checkpoint, directory, dimension, config, *, fixed_dmrg_policy=None
):
    """One explicit fixed-MO protocol, used for surveys and terminal evidence."""
    import resource

    from .solvers.block2_convergence import diagnostic_config

    _check_dmrg_states(context.states, partition["electrons"], optimize_orbitals=False)
    options = {}
    if fixed_dmrg_policy is not None:
        from .solvers.block2_convergence import native_diagnostic_config, validate_fixed_policy

        fixed_dmrg_policy = validate_fixed_policy(fixed_dmrg_policy)
        options = dict(fixed_dmrg_policy=fixed_dmrg_policy)
    input_binding = dict(
        basis_id=context.basis_id,
        partition=partition,
        checkpoint=binding(checkpoint),
        dimension=dimension,
        config=config,
        **options,
    )
    directory = Path(directory).resolve()
    cached = directory / "fixed_result.json"
    if cached.exists():
        result = read_json(cached)
        if result.pop("result_id") != identity(result):
            raise ContractError("Fixed-MO result content changed")
        if result["input_binding"] != input_binding:
            raise ContractError("Fixed-MO result identity differs")
        for key in ("qit_artifact", "integral_artifact"):
            if key in result:
                verify_binding(result[key])
        for item in result.get("convergence_artifacts", {}).values():
            verify_binding(item)
        for key in ("densities", "entropies", "mutual_information"):
            if result.get(key) is not None:
                result[key] = np.asarray(result[key])
        return result
    _allow_dispatch()
    directory = checked_directory(directory, config["threads"])
    convergence_artifacts = {}

    def completed(result):
        result["input_binding"] = input_binding
        if fixed_dmrg_policy is not None:
            result["convergence_artifacts"] = convergence_artifacts
        write_json(cached, dict(result, result_id=identity(result)))
        return result

    export_options = {} if fixed_dmrg_policy is None else dict(dimension=dimension, **options)
    export = _export_integrals(
        context, partition, checkpoint, directory / "integral_export", config, **export_options
    )
    work = directory / "dmrg"
    work.mkdir()
    shutil.copy2(export["integrals"]["path"], work / "FCIDUMP")
    ctx = dict(
        electrons=partition["electrons"],
        states=context.states,
        spin=context.states["spin_multiplicity"] - 1,
    )
    if fixed_dmrg_policy is None:
        text = diagnostic_config(
            ctx, dimension, "standard", config["dmrg_max_sweeps"], threads=config["threads"]
        )
        gate_request = dict(profile="standard")
    else:
        native_path = verify_binding(export["native_configuration"])
        text, native_config = native_diagnostic_config(
            native_path, ctx, dimension, config["dmrg_max_sweeps"], threads=config["threads"]
        )
        gate_request = dict(fixed_dmrg_policy=fixed_dmrg_policy, native_config=native_config)
        convergence_artifacts["native_configuration"] = export["native_configuration"]
    (work / "dmrg.conf").write_text(text)
    write_json(work / "state_sweep_gate_request.json", gate_request)
    if fixed_dmrg_policy is not None:
        convergence_artifacts.update(
            configuration=binding(work / "dmrg.conf"),
            request=binding(work / "state_sweep_gate_request.json"),
        )
    env = {
        **os.environ,
        **_dmrg_environment(),
        **{
            k: str(config["threads"])
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    start = time.perf_counter()
    with (work / "dmrg.out").open("w") as log:
        job = subprocess.run(
            [sys.executable, "-m", "pas.solvers.diagnostic_sweep_gate", "dmrg.conf"],
            cwd=work,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    timing = dict(
        cpu_seconds=export["cpu_seconds"]
        + after.ru_utime
        + after.ru_stime
        - before.ru_utime
        - before.ru_stime,
        wall_seconds=export["wall_seconds"] + time.perf_counter() - start,
        return_code=job.returncode,
    )
    write_json(directory / "execution.json", timing)
    gate = (
        read_json(work / "state_sweep_gate.json")
        if (work / "state_sweep_gate.json").exists()
        else None
    )
    if job.returncode and (fixed_dmrg_policy is None or job.returncode != 96):
        raise RuntimeError(
            "Block2 process failed outside numerical convergence: " + str(work / "dmrg.out")
        )
    if fixed_dmrg_policy is None:
        if not gate or not gate["passed"]:
            return completed(dict(valid=False, failure="DMRG_state_sweep_gate_failed", **timing))
    else:
        for item in convergence_artifacts.values():
            verify_binding(item)
        passed = _check_native_fixed_gate(
            gate, fixed_dmrg_policy, native_config, len(context.states["roots"]), job.returncode
        )
        convergence_artifacts["endpoint"] = binding(work / "state_sweep_gate.json")
        if not passed:
            return completed(
                dict(valid=False, failure="DMRG_all_root_endpoint_nonconvergence", **timing)
            )
    sweeps = np.asarray(np.load(work / "node0/sweep_energies.npy"), float)
    if (
        sweeps.ndim != 2
        or sweeps.shape[1] != len(context.states["roots"])
        or len(sweeps) < 2
        or not np.all(np.isfinite(sweeps))
    ):
        raise ContractError("Malformed DMRG per-root sweep energies")
    deltas = abs(sweeps[-1] - sweeps[-2])
    if fixed_dmrg_policy is None:
        if np.max(deltas) > 1e-8:
            return completed(
                dict(valid=False, failure="DMRG_state_energy_nonconvergence", **timing)
            )
    elif np.max(deltas) >= native_config["sweep_tol"] or not np.allclose(
        deltas, gate["stages"][-1]["all_root_deltas"], atol=1e-12, rtol=0
    ):
        raise ContractError("Saved sweep energies differ from the native all-root endpoint")
    data = _extract_mps(work, context, partition, directory / "evidence_extract", config)
    timing["cpu_seconds"] += data["cpu_seconds"]
    timing["wall_seconds"] += data["wall_seconds"]
    return completed(
        dict(
            valid=True,
            **timing,
            **{k: data[k] for k in ("densities", "entropies", "mutual_information")},
            root_energies=data["result"]["root_energies"],
            qit_artifact=data["artifact"],
            integral_artifact=binding(directory / "integral_export/integrals.json"),
            provenance=dict(
                basis_id=context.basis_id,
                checkpoint=binding(checkpoint),
                integrals=export["integrals"],
                bond_dimension=dimension,
                scope=partition,
                orbital_gauge=context.payload.get("gauge"),
                energy_source=data["result"]["energy_source"],
                energy_density_gaps=data["result"]["energy_density_gaps"],
                finite_m_error_certified=False,
                checkpoint_identity=data["result"]["checkpoint"]["identity"],
            ),
        )
    )


def survey_quadratic(context, partition, densities, entropies, mutual_information, config):
    """Proposal only: fixed-scope restricted statistics, never target-CAS labels."""
    config = configuration(config)
    from .preferences import assert_preference_binding

    assert_preference_binding(context, config)
    if config.get("survey_model") == "posterior_evidence_capture_v1":
        from .evidence_belief import posterior_capture

        return posterior_capture(
            context, partition, densities, entropies, mutual_information, config
        )
    k = context.k
    n = context.nmo
    active = partition["active"]
    sw = np.asarray(context.states["weights"])
    if "survey_model" in config:
        # The caller must restrict candidates to these measured indices. Zero
        # coefficients outside this scope encode no knowledge of those orbitals.
        model = config["survey_model"]
        m = len(active)
        roots = len(context.states["roots"])
        if (
            context.backend != "dmrg"
            or k < 2
            or m < k
            or len(set(active)) != m
            or any(type(i) is not int for i in active)
            or not set(active) <= set(context.window)
        ):
            raise ContractError(
                "Survey model requires a complete, distinct measured DMRG scope and K >= 2"
            )
        s = np.asarray(entropies, float)
        mi = np.asarray(mutual_information, float)
        if s.shape != (roots, m) or mi.shape != (roots, m, m):
            raise ContractError("Survey QIT arrays do not cover every root and measured orbital")
        metrics.qit(s, mi, sw)
        if not np.allclose(mi, mi.swapaxes(1, 2), atol=1e-12, rtol=0) or np.any(
            np.diagonal(mi, axis1=1, axis2=2) != 0
        ):
            raise ContractError(
                "Survey mutual information must be symmetric with an exact zero diagonal"
            )
        if "occupation" in config["channels"]:
            electrons = partition.get("electrons")
            if type(electrons) is not int or not 0 <= electrons <= 2 * m:
                raise ContractError("Survey density requires its explicit physical electron scope")
            densities = metrics.validate_densities(densities, sw, electrons)
            if densities.shape != (roots, m, m):
                raise ContractError("Survey density does not cover every root and measured orbital")
        if "composition" in config["channels"]:
            projection = np.asarray(context.composition, float)
            if (
                projection.shape != (n,)
                or not np.all(np.isfinite(projection))
                or np.any(projection < 0)
                or np.any(projection > 1)
            ):
                raise ContractError(
                    "Survey composition requires complete finite AO projections in [0, 1]"
                )
        ent = sw @ s
        mutual = np.einsum("r,rij->ij", sw, mi)
        if model == "correlation_cut_v1":
            entropy_scale = k * np.log(4)
            pair_scale = k * (k - 1) / 2 * np.log(4)
            qit_weight = 1.0
            score_channels = ["qit"]
            normalization = dict(
                entropy=float(entropy_scale),
                mutual_information=float(pair_scale),
                aggregation="state_weighted_before_normalization",
            )
        else:
            entropy_scale = float(ent.sum())
            pair_scale = float(mutual.sum() / 2)
            projection_scale = float(projection[active].sum())
            if not all(np.isfinite(v) and v > 0 for v in (entropy_scale, pair_scale)):
                raise ContractError(
                    "Evidence capture requires positive measured entropy and MI denominators"
                )
            qit_weight = config["weights"]["qit"] / sum(config["weights"].values())
            score_channels = list(config["channels"])
            normalization = dict(
                entropy=entropy_scale,
                mutual_information=pair_scale,
                projection=projection_scale,
                aggregation="state_weighted_before_normalization",
            )
        linear = np.zeros(n)
        quad = np.zeros((n, n))
        # cut(A)=sum_i x_i degree_i - 2*internal(A); each undirected MI
        # edge is counted once, while xQx visits both matrix triangles.
        linear[active] = qit_weight * 0.5 * (ent / entropy_scale - mutual.sum(axis=1) / pair_scale)
        quad[np.ix_(active, active)] = qit_weight * 0.75 * mutual / pair_scale
        if model == "evidence_capture_v1":
            total = sum(config["weights"].values())
            weight = config["weights"]["occupation"] / total
            diag = np.diagonal(densities, axis1=1, axis2=2).real
            linear[active] += weight * (sw @ (2 * diag - diag**2)) / k
            pair = -weight * np.einsum("r,rij->ij", sw, np.abs(densities) ** 2) / k
            np.fill_diagonal(pair, 0.0)
            quad[np.ix_(active, active)] += pair
            if context.composition_preferences is None:
                linear[active] += config["weights"]["composition"] / total * projection[active] / k
            else:
                component = metrics.composition_quadratic(
                    context.composition,
                    context.active,
                    context.composition_pair,
                    config["composition_model"],
                    composition_preferences=context.composition_preferences,
                )
                weight = config["weights"]["composition"] / total
                linear[active] += weight * component["linear"][active]
                quad[np.ix_(active, active)] += weight * component["pair"][np.ix_(active, active)]
        result = dict(
            constant=0.0,
            linear=linear.tolist(),
            quadratic=quad.tolist(),
            quadratic_convention="xQx",
            scope="complete_measured_window_statistics_only_not_target_CAS_truth",
            score_model=model,
            measured_indices=list(active),
            candidate_scope="must_be_subset_of_measured_indices",
            score_channels=score_channels,
            evidence_channels=list(config["channels"]),
            cut_penalty=1.0,
            normalization=normalization,
            score_definition=(
                "0.5*(S_A/(K*ln4)+(MI_internal-MI_cut)/(choose(K,2)*ln4))"
                if model == "correlation_cut_v1"
                else "weighted_mean(Un,P_A/K,0.5*(S_A/S_W+(MI_internal-MI_cut)/MI_W))"
            ),
            occupation_statistic="restricted_full_1RDM_2TrD_minus_TrD2_over_K",
            mutual_information_convention="half_mutual_information_natural_log",
        )
        if context.composition_preferences is not None:
            from .evidence_belief import _composition_descriptor

            result["composition_descriptor"], _ = _composition_descriptor(
                context, config["composition_model"], active
            )
            result["score_definition"] = (
                "weighted_mean(Un,terminal_composition_with_preferences,0.5*(S_A/S_W+(MI_internal-MI_cut)/MI_W))"
            )
        return result
    linear = np.zeros(n)
    quad = np.zeros((n, n))
    total = sum(config["weights"].values())
    for channel in config["channels"]:
        weight = config["weights"][channel] / total
        if channel == "occupation":
            diag = np.diagonal(densities, axis1=1, axis2=2)
            linear[active] += weight * (sw @ (2 * diag - diag**2)) / k
            pair = -weight * np.einsum("r,rij->ij", sw, np.abs(densities) ** 2) / k
            np.fill_diagonal(pair, 0.0)
            quad[np.ix_(active, active)] += pair
        elif channel == "composition":
            component = metrics.composition_quadratic(
                context.composition,
                context.active,
                context.composition_pair,
                config["composition_model"],
                composition_channels=context.composition_channels,
                composition_preferences=context.composition_preferences,
                composition_purity_weight=config.get("composition_purity_weight", 0.5),
            )
            linear += weight * component["linear"]
            quad += weight * component["pair"]
        else:
            factor = 0.5 if k > 1 else 1.0
            linear[active] += weight * factor * (sw @ entropies) / (k * np.log(4))
            if k > 1:
                quad[np.ix_(active, active)] += (
                    weight
                    * 0.25
                    * np.einsum("r,rij->ij", sw, mutual_information)
                    / (k * (k - 1) / 2 * np.log(4))
                )
    return dict(
        constant=0.0,
        linear=linear.tolist(),
        quadratic=quad.tolist(),
        quadratic_convention="xQx",
        scope="fixed_survey_restricted_statistics_not_target_CAS_truth",
    )


def _fixed_dmrg_output_context(context, partition, data, directory, config):
    """Bind the new partition and final MPS density without rotating any MO.

    This is an explicitly derived orbital container, not a Molcas CI output.
    Its state metadata come only from the bound fixed-MO evidence artifact.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    old = context.payload
    mo = read_openmolcas_orbital_hdf5(old["basis_file"])
    densities = np.asarray(data["densities"], float)
    metrics.validate_densities(densities, context.states["weights"], partition["electrons"])
    energies = np.asarray(data["root_energies"], float)
    if energies.shape != (len(context.states["roots"]),) or not np.isfinite(energies).all():
        raise ContractError("Fixed DMRG output lacks complete finite root energies")
    evidence = data["qit_artifact"]
    verify_binding(evidence)
    request = dict(
        source_basis=binding(old["basis_file"]),
        source_checkpoint=binding(old["checkpoint"]),
        partition=partition,
        mps_evidence=evidence,
        config_id=identity(config),
        final_state_data_id=identity(
            dict(densities=densities.tolist(), root_energies=energies.tolist())
        ),
    )
    record_path = directory / "fixed_context.json"
    if record_path.exists():
        saved = read_json(record_path)
        if any(saved.get(key) != value for key, value in request.items()):
            raise ContractError("Derived fixed DMRG context request changed")
        basis = verify_binding(saved["basis_file"])
        checkpoint = verify_binding(saved["checkpoint"])
        output = context_from_files(
            old["case"],
            basis,
            checkpoint,
            config,
            electrons=partition["electrons"],
            reference_context=context,
            input_partition=partition,
        )
        if not np.array_equal(
            read_openmolcas_orbital_hdf5(basis).mo_coefficients, mo.mo_coefficients
        ):
            raise ContractError("Cached fixed DMRG orbital container changed the input MOs")
        mapping = _audit_output(context, output, partition, "probe")
        if mapping != saved["basis_mapping"]:
            raise ContractError("Derived fixed DMRG orbital mapping changed")
        output.payload["gauge"] = "fixed_input_MO_basis_v1"
        return output, mapping, binding(record_path)
    tags = [""] * context.nmo
    occupations = np.zeros(context.nmo)
    for key, tag in [
        ("frozen", "f"),
        ("inactive", "i"),
        ("active", "2"),
        ("secondary", "s"),
        ("deleted", "d"),
    ]:
        for index in partition[key]:
            tags[index] = tag
    occupations[partition["frozen"] + partition["inactive"]] = 2.0
    occupations[partition["active"]] = np.diag(
        np.einsum("r,rij->ij", context.states["weights"], densities)
    )
    checkpoint = directory / "fixed.RasOrb"
    basis = directory / "fixed.rasscf.h5"
    text = _replace_type_indices(Path(old["checkpoint"]).read_text(), tags)
    records = "".join(
        "".join(" " + format(float(v), "21.14E") for v in occupations[start : start + 5]) + "\n"
        for start in range(0, len(occupations), 5)
    )
    checkpoint.write_text(
        _replace_orbital_section(
            text, "OCC", "* FIXED MPS STATE-AVERAGE DIAGONAL OCCUPATIONS\n" + records
        )
    )
    native, _ = read_rasorb_coefficients(
        old["checkpoint"], symmetry_block_sizes=mo.symmetry_block_sizes
    )
    proof = verify_native_inporb22(checkpoint, native)
    shutil.copy2(old["basis_file"], basis)
    with h5py.File(basis, "r+") as h:
        h["MO_OCCUPATIONS"][...] = occupations
        h["MO_TYPEINDICES"][...] = np.asarray(tags, dtype="S1")
        for name, value in [("ROOT_ENERGIES", energies), ("DENSITY_MATRIX", densities)]:
            if name in h:
                del h[name]
            h.create_dataset(name, data=value)
        for name in (
            "CI_VECTORS",
            "SPINDENSITY_MATRIX",
            "TRANSITION_DENSITY_MATRIX",
            "TRANSITION_SPIN_DENSITY_MATRIX",
        ):
            if name in h:
                del h[name]
        for name, value in dict(
            NACTEL=partition["electrons"],
            SPINMULT=context.states["spin_multiplicity"],
            NROOTS=len(energies),
            NSTATES=len(energies),
            STATE_ROOTID=context.states["roots"],
            STATE_WEIGHT=context.states["weights"],
        ).items():
            h.attrs[name] = value
        h.attrs["PAS_ORBITAL_CONTAINER_ROLE"] = "derived_fixed_MO_partition_and_final_MPS_density"
        h.attrs["PAS_FIXED_MPS_EVIDENCE_SHA256"] = evidence["sha256"]
    output = context_from_files(
        old["case"],
        basis,
        checkpoint,
        config,
        electrons=partition["electrons"],
        reference_context=context,
        input_partition=partition,
    )
    derived = read_openmolcas_orbital_hdf5(basis)
    if derived.type_indices != tuple(tags) or not np.array_equal(
        derived.mo_coefficients, mo.mo_coefficients
    ):
        raise ContractError("Derived fixed DMRG orbital container changed the input MOs")
    mapping = _audit_output(context, output, partition, "probe")
    output.payload["gauge"] = "fixed_input_MO_basis_v1"
    record = dict(
        schema="pas_fixed_dmrg_output_context_v1",
        **request,
        basis_file=binding(basis),
        checkpoint=binding(checkpoint),
        orbital_optimization=False,
        orbital_coefficients_unchanged=True,
        native_fixed_record_proof=proof,
        basis_mapping=mapping,
    )
    write_json(record_path, record)
    return output, mapping, binding(record_path)


def _execute_dmrg(
    context,
    partition,
    kind,
    directory,
    config,
    request,
    *,
    fixed_dmrg_policy=None,
    geometry_transition=None,
):
    runtime_identity = runtime()
    artifact = dict(request=binding(directory / "request.json"))
    timing = dict(cpu_seconds=0.0, wall_seconds=0.0, return_code=0)
    output = context
    final_partition = partition
    mapping = None
    qn = None
    data = None
    optimize_orbitals = kind == "optimize" and config.get("orbital_optimization") is not False
    if optimize_orbitals:
        from .solvers.molcas_convergence import install

        native_record = directory / "native_execution.json"
        if native_record.exists():
            timing = read_json(native_record)
        else:
            _allow_dispatch()
            env = install(directory, _dmrg_environment(), config["dmrg_max_sweeps"])
            timing = run_solver(directory, request["input"], config["threads"], environment=env)
            write_json(native_record, timing)
        log = (directory / "solver.log").read_text(errors="replace")
        failed_endpoints = [
            read_json(p)
            for p in sorted((directory / "block2_calls").glob("call_*/sweep_endpoint.json"))
        ]
        numerical_endpoint = any(
            any(
                stage.get("failure", "").startswith("NumericalNonconvergence:")
                for stage in item.get("stages", [])
            )
            for item in failed_endpoints
        )
        if numerical_endpoint:
            process_records = [
                read_json(p)
                for p in sorted((directory / "block2_calls").glob("call_*/process.json"))
            ]
            last_codes = re.findall(
                r"---\s*Stop Module:\s*rasscf\b[^\n]*/rc=(_RC_[A-Z_]+_)", log, re.I
            )
            expected = {96: "_RC_NOT_CONVERGED_", 130: "_RC_GENERAL_ERROR_"}
            if (
                timing["return_code"] not in expected
                or not last_codes
                or last_codes[-1].upper() != expected[timing["return_code"]]
                or not process_records
                or process_records[-1].get("return_code") != 96
                or re.search(
                    r"No space left|Permission denied|Segmentation fault|command not found|Input/output error",
                    log,
                    re.I,
                )
            ):
                raise RuntimeError(
                    "DMRGSCF infrastructure failure after numerical endpoint failure"
                )
            return dict(
                request_id=request["request_id"],
                context_id=context.context_id,
                kind=kind,
                active=partition["active"],
                valid=False,
                converged=False,
                state_valid=False,
                metrics={},
                failure="DMRG_all_root_endpoint_nonconvergence",
                **timing,
                artifacts={**artifact, "log": binding(directory / "solver.log")},
            )
        if not _converged(log, kind, timing["return_code"]):
            return dict(
                request_id=request["request_id"],
                context_id=context.context_id,
                kind=kind,
                active=partition["active"],
                valid=False,
                converged=False,
                state_valid=False,
                metrics={},
                failure="DMRGSCF_numerical_nonconvergence",
                **timing,
                artifacts={**artifact, "log": binding(directory / "solver.log")},
            )
        qn = verify_qn(log)
        calls = sorted((directory / "block2_calls").glob("call_*/audit.json"))
        if not calls:
            raise ContractError("No audited Block2 optimization calls")
        for call in calls:
            if (
                max(read_json(call)["settings"]["schedule"][0]) != config["dmrg_optimize_m"]
                or not read_json(call.parent / "sweep_endpoint.json")["passed"]
            ):
                raise ContractError("Optimizer Block2 fidelity/endpoint contract differs")
            if not (call.parent / "energy_density.json").exists():
                raise ContractError("Missing final-density energy binding")
            endpoint = read_json(call.parent / "sweep_endpoint.json")
            if any(
                "all_root_deltas" not in stage
                or len(stage["all_root_deltas"]) != len(context.states["roots"])
                or max(stage["all_root_deltas"]) >= stage["native_tolerance"]
                for stage in endpoint["stages"]
            ):
                raise ContractError("Optimizer did not converge all roots at the density endpoint")
        manifest = {
            str(p.relative_to(directory)): file_hash(p)
            for call in calls
            for p in (
                call,
                call.parent / "sweep_endpoint.json",
                call.parent / "energy_density.json",
                call.parent / "original.conf",
                call.parent / "effective.conf",
                call.parent / "process.json",
            )
            if p.exists()
        }
        write_json(directory / "native_calls_manifest.json", manifest)
        artifact["native_calls"] = binding(directory / "native_calls_manifest.json")
        if geometry_transition is None:
            output = context_from_files(
                context.payload["case"],
                directory / "job.rasscf.h5",
                directory / "job.RasOrb",
                config,
                electrons=partition["electrons"],
                reference_context=context,
                input_partition=partition,
            )
        else:
            from .geometry_transfer import output_context

            output = output_context(
                context,
                partition,
                directory / "job.rasscf.h5",
                directory / "job.RasOrb",
                config,
                geometry_transition,
            )
        if "qit" in config["channels"]:
            # Only QIT depends on a particular within-active orbital gauge.
            # The gradient MPS has no proven binding to Molcas's last natural
            # rotation, so buy one explicit fixed-MO evidence stage, charged
            # to this optimize request. No-QIT masks need no additional CI.
            if _RECONCILE_ONLY.get():
                gauge = directory / "canonical_gauge/gauge.json"
                if not gauge.exists():
                    raise RuntimeError("Unresolved canonical orbital gauge stage")
                record = read_json(gauge)
                if record["source_basis"] != binding(output.payload["basis_file"]) or record[
                    "source_checkpoint"
                ] != binding(output.payload["checkpoint"]):
                    raise ContractError("Saved canonical gauge source binding differs")
                output = context_from_files(
                    output.payload["case"],
                    gauge.parent / "canonical.rasscf.h5",
                    gauge.parent / "canonical.RasOrb",
                    config,
                    electrons=partition["electrons"],
                    reference_context=output,
                    input_partition=output.partition(output.active),
                )
                output.payload["gauge"] = record["name"]
                gauge_artifact = binding(gauge)
            else:
                output, gauge_artifact = canonicalize_dmrg_context(
                    output, directory / "canonical_gauge", config
                )
            artifact["gauge"] = gauge_artifact
            for name, item in read_json(gauge_artifact["path"])["density_source"][
                "sources"
            ].items():
                artifact["native_density_" + name] = item
        if geometry_transition is None:
            mapping = _audit_output(context, output, partition, kind)
        else:
            from .geometry_transfer import audit_output

            mapping = audit_output(context, output, partition, geometry_transition, directory)
        final_partition = output.partition(output.active)
        _energy_check(directory / "job.rasscf.h5", context.states)
        checkpoint = output.payload["checkpoint"]
        dimension = config["dmrg_optimize_m"]
        if "qit" not in config["channels"]:
            densities = None
            if "occupation" in config["channels"]:
                densities, energies, density_source = read_dmrg_densities(output)
                for name, item in density_source["sources"].items():
                    artifact["native_density_" + name] = item
            else:
                with h5py.File(directory / "job.rasscf.h5") as h:
                    energies = np.asarray(h["ROOT_ENERGIES"][()], float).reshape(-1)
            final_energy = read_json(calls[-1].parent / "energy_density.json")
            expected = np.asarray([s["energy"] for s in final_energy["states"]])
            if not np.allclose(energies, expected, atol=1e-7, rtol=0):
                raise ContractError(
                    "Native output energy is not the audited final gradient density energy"
                )
            data = dict(
                valid=True,
                cpu_seconds=0.0,
                wall_seconds=0.0,
                densities=densities,
                entropies=None,
                mutual_information=None,
                root_energies=energies.tolist(),
                provenance=dict(
                    basis_id=output.basis_id,
                    scope=final_partition,
                    energy_source="audited_final_density_supplied_to_Molcas",
                    density_gauge="verified_native_output_MO_basis"
                    if densities is not None
                    else None,
                    density_source=density_source if densities is not None else None,
                    terminal_fixed_MPS_required=False,
                ),
            )
            artifact["gradient_energy_density"] = binding(calls[-1].parent / "energy_density.json")
    else:
        if (directory / "started.json").exists():
            if not _RECONCILE_ONLY.get():
                raise RuntimeError("Unreconciled DMRG action; automatic rerun is forbidden")
        else:
            _allow_dispatch()
            write_json(directory / "started.json", dict(pid=os.getpid(), started_unix=time.time()))
        checkpoint = directory / "trial.RasOrb"
        dimension = config["dmrg_optimize_m"] if kind == "optimize" else config["dmrg_probe_m"]
        mapping = None
    if data is None:
        options = {} if fixed_dmrg_policy is None else dict(fixed_dmrg_policy=fixed_dmrg_policy)
        data = _fixed_dmrg(
            output,
            final_partition,
            checkpoint,
            directory / "terminal_evidence",
            dimension,
            config,
            **options,
        )
    artifact.update(
        {"fixed_" + key: value for key, value in data.get("convergence_artifacts", {}).items()}
    )
    timing["cpu_seconds"] += data["cpu_seconds"]
    timing["wall_seconds"] += data["wall_seconds"]
    result = dict(
        request_id=request["request_id"],
        context_id=context.context_id,
        kind=kind,
        active=partition["active"],
        valid=False,
        converged=False,
        state_valid=False,
        metrics={},
        **timing,
        artifacts=artifact,
        runtime=runtime_identity,
    )
    if not data["valid"]:
        result["failure"] = data["failure"]
        write_json(directory / "execution.json", timing)
        return result
    if kind == "optimize" and not optimize_orbitals:
        output, mapping, fixed_context = _fixed_dmrg_output_context(
            context, partition, data, directory / "fixed_context", config
        )
        artifact["fixed_context"] = fixed_context
        final_partition = output.partition(output.active)
    values = metrics.evaluate(
        config["channels"],
        config["weights"],
        densities=data["densities"],
        state_weights=context.states["weights"],
        electrons=final_partition["electrons"],
        composition=output.composition,
        active=final_partition["active"],
        entropies=data["entropies"],
        mutual_information=data["mutual_information"],
        composition_pair=output.composition_pair,
        composition_channels=output.composition_channels,
        composition_model=config["composition_model"],
        composition_preferences=output.composition_preferences,
        composition_purity_weight=config.get("composition_purity_weight", 0.5),
    )
    values["energy"] = float(np.asarray(data["root_energies"]) @ context.states["weights"])
    if data["densities"] is not None:
        np.save(directory / "densities.npy", data["densities"])
        artifact["densities"] = binding(directory / "densities.npy")
    if "qit_artifact" in data:
        artifact.update(mps_evidence=data["qit_artifact"], integrals=data["integral_artifact"])
    if kind == "optimize":
        artifact.update(
            basis_file=binding(output.payload["basis_file"]),
            checkpoint=binding(output.payload["checkpoint"]),
        )
        if optimize_orbitals:
            artifact["log"] = binding(directory / "solver.log")
        result["output_context"] = output.to_dict()
    result.update(
        valid=True,
        converged=True,
        state_valid=True,
        metrics=values,
        provenance=dict(
            **data["provenance"],
            basis_mapping=mapping,
            qn=qn,
            terminal_evidence_role=(
                "post_optimization_fixed_MO"
                if "qit" in config["channels"]
                else "native_optimization_output"
            )
            if optimize_orbitals
            else "fixed_input_MO",
            terminal_evidence_cost_included=True,
            energy_used_for_reward=False,
        ),
    )
    if config.get("orbital_optimization") is False:
        result["provenance"].update(orbital_optimization=False, method="DMRGCI")
    if kind == "survey":
        result["proposal_quadratic"] = survey_quadratic(
            context,
            partition,
            data["densities"],
            data["entropies"],
            data["mutual_information"],
            config,
        )
        result["target_response_label"] = False
    write_json(directory / "execution.json", timing)
    artifact["execution"] = binding(directory / "execution.json")
    return result
