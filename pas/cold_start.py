"""Reusable physical SCF -> initial optimization, independent of search state."""

from __future__ import annotations

import shutil
from pathlib import Path

import h5py
import numpy as np

from .common import file_hash, identity, read_json, write_json
from .config import configuration


def physical_case(case, *, composition=True, active_space=None):
    """Allowlist chemistry only; archived policy settings never enter a Context."""
    import copy

    allowed = (
        "key",
        "case_id",
        "geometry_xyz",
        "basis",
        "basis_library",
        "basis_by_element",
        "cholesky",
        "charge",
        "relativistic",
        "group",
        "states",
        "active_space",
    )
    result = {name: copy.deepcopy(case[name]) for name in allowed if name in case}
    result.setdefault("charge", 0)
    if active_space is not None:
        result["active_space"] = copy.deepcopy(active_space)
    if composition and "target" in case:
        result["target"] = copy.deepcopy(case["target"])
    return result


def from_verified_predecessor(seed, seed_path, config, output, reason, scf_source=None):
    """Complete missing initialization only from an explicitly bound neutral SCF."""
    from . import backend

    candidate = (
        scf_source or seed.get("neutral_scf") or seed.get("initialization", {}).get("source_scf")
    )
    if candidate is None:
        raise backend.ContractError(
            reason + "; initialize from neutral SCF (no verified predecessor receipt supplied)"
        )
    if isinstance(candidate, dict):
        candidate = backend.verify_binding(candidate)
    else:
        candidate = Path(candidate)
    source = read_json(candidate)
    if source.get("role") != "neutral_scf" or source.get("converged") is not True:
        raise backend.ContractError("Predecessor is not a verified neutral SCF receipt")
    original = (
        seed.get("initialization", {}).get("source_case")
        or seed.get("context", {}).get("payload", {}).get("case")
        or seed.get("snapshot", {}).get("case")
    )
    if original is None:
        raise backend.ContractError("Missing physical case for initialization predecessor")
    budget = seed.get("snapshot", {}).get("budget")
    active_space = {"electrons": budget[0], "orbitals": budget[1]} if budget else None
    case = physical_case(original, active_space=active_space)
    record = dict(
        source_seed=backend.binding(seed_path),
        source_scf=backend.binding(candidate),
        reason=reason,
        reported_source_initialization_protocol=seed.get("initialization_protocol"),
    )
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    record_path = destination / "predecessor_fallback.json"
    if record_path.exists() and read_json(record_path) != record:
        raise backend.ContractError("Initialization predecessor changed")
    write_json(record_path, record)
    return prepare_cold_start(case, destination, config, candidate)


def frontier_partition(occupations, energies, type_indices, electrons, orbitals):
    """Neutral closed-shell initial guess; never a restriction on later candidates."""
    occupations, energies = np.asarray(occupations, float), np.asarray(energies, float)
    if occupations.shape != energies.shape or not np.isfinite([occupations, energies]).all():
        raise ValueError("Invalid SCF occupations/energies")
    if (
        electrons % 2
        or electrons < 0
        or not 0 < orbitals <= len(occupations)
        or electrons > 2 * orbitals
    ):
        raise ValueError("Invalid closed-shell initial active space")
    eligible = [i for i, t in enumerate(type_indices) if t not in ("f", "d")]
    occupied = [i for i in eligible if abs(occupations[i] - 2) < 1e-6]
    virtual = [i for i in eligible if abs(occupations[i]) < 1e-6]
    if len(occupied) + len(virtual) != len(eligible):
        raise ValueError("A neutral SCF initial guess requires integer closed-shell occupations")
    filled, empty = electrons // 2, orbitals - electrons // 2
    if len(occupied) < filled or len(virtual) < empty:
        raise ValueError("SCF lacks the required occupied/empty seed orbitals")
    active = sorted(
        sorted(occupied, key=lambda i: (-energies[i], i))[:filled]
        + sorted(virtual, key=lambda i: (energies[i], i))[:empty]
    )
    return active, sorted(i for i in occupied if i not in active)


def prepare_cold_start(case, directory, config, scf_source=None):
    import fcntl

    from . import backend

    config = configuration(config)
    source_case = physical_case(case)
    case = physical_case(case, composition="composition" in config["channels"])
    directory = backend.checked_directory(directory, config["threads"])
    with (directory / "initialization.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        backend.check_initial_capabilities(case, config)
        if case["active_space"]["electrons"] % 2 or case["states"]["spin_multiplicity"] != 1:
            raise ValueError("Cold-start selector currently supports closed-shell singlet seeds")
        return _prepare_cold_start(case, directory, config, scf_source, source_case)


def _prepare_cold_start(case, directory, config, scf_source=None, source_case=None):
    from . import backend

    cached = directory / "seed.json"
    request = dict(
        case=case,
        config=config,
        scf_source=(
            dict(path=str(Path(scf_source).resolve()), sha256=file_hash(scf_source))
            if scf_source
            else None
        ),
        initial_selector="neutral_energy_frontier_v1",
    )
    request["request_id"] = identity(request)
    if cached.exists():
        if read_json(directory / "initial_request.json") != request:
            raise ValueError("Initialization cache identity differs")
        return backend.import_seed(cached, config, directory)
    if (directory / "initial_request.json").exists() and read_json(
        directory / "initial_request.json"
    ) != request:
        raise ValueError("An initialization with different settings already exists")
    write_json(directory / "initial_request.json", request)
    scf = directory / "scf"
    scf.mkdir(exist_ok=True)
    if scf_source is not None:
        # Only a hash-bound neutral SCF bundle is accepted; optimized or AVAS seeds
        # cannot be disguised as an evidence-independent mean-field foundation.
        source = read_json(scf_source)
        if source.get("role") != "neutral_scf" or source.get("converged") is not True:
            raise ValueError("SCF reuse requires a converged neutral_scf receipt")
        if source.get("initial_selection_used_composition") not in (None, False):
            raise ValueError("SCF reuse cannot contain composition-selected orbitals")
        if source.get("physical_case_id") != identity(
            {k: case[k] for k in ("geometry_xyz", "basis", "charge", "relativistic")}
        ):
            raise ValueError("SCF physical input changed")
        for key, name in (("basis_file", "job.scf.h5"), ("checkpoint", "job.ScfOrb")):
            backend.verify_binding(source[key])
            backend._copy_artifact(source[key]["path"], scf / name)
        timing = dict(
            cpu_seconds=0.0,
            wall_seconds=0.0,
            reused=True,
            historical_cpu_seconds=source.get("historical_cpu_seconds", source.get("cpu_seconds")),
            historical_wall_seconds=source.get(
                "historical_wall_seconds", source.get("wall_seconds")
            ),
        )
    else:
        (scf / "geometry.xyz").write_text(case["geometry_xyz"])
        text = (
            backend.gateway_input(case)
            + backend.seward_input(case, "cas")
            + f"&SCF\nCharge={case.get('charge', 0)}\n"
        )
        timing = (
            read_json(scf / "execution.json")
            if (scf / "execution.json").exists()
            else backend.run_solver(scf, text, config["threads"])
        )
        log = (scf / "solver.log").read_text(errors="replace").lower()
        if timing["return_code"] or "happy landing" not in log:
            raise RuntimeError("Cold-start SCF did not finish successfully; no automatic retry")
    basis, checkpoint = scf / "job.scf.h5", scf / "job.ScfOrb"
    mo = backend.read_openmolcas_orbital_hdf5(basis)
    inp = backend.read_inporb(checkpoint)
    if set(inp.type_indices) - {"f", "i", "s", "d"}:
        raise ValueError("Neutral SCF receipt contains correlated active orbitals")
    coefficients, tags = backend.read_rasorb_coefficients(
        checkpoint, symmetry_block_sizes=mo.symmetry_block_sizes
    )
    if (
        mo.type_indices != inp.type_indices
        or tags != inp.type_indices
        or not np.allclose(mo.occupations, inp.occupations, atol=1e-6, rtol=0)
        or not np.allclose(
            mo.native_coefficients_to_ao(coefficients), mo.mo_coefficients, atol=1e-8, rtol=0
        )
    ):
        raise ValueError("SCF HDF5/checkpoint mismatch")
    if not np.allclose(
        mo.mo_coefficients.T @ mo.ao_overlap @ mo.mo_coefficients,
        np.eye(mo.orbital_count),
        atol=1e-7,
        rtol=0,
    ):
        raise ValueError("SCF orbitals are not AO-metric orthonormal")
    backend._geometry(case, mo, basis)
    neutral = dict(
        role="neutral_scf",
        converged=True,
        physical_case_id=identity(
            {k: case[k] for k in ("geometry_xyz", "basis", "charge", "relativistic")}
        ),
        basis_file=backend.binding(basis),
        checkpoint=backend.binding(checkpoint),
        initial_selection_used_composition=False,
        **timing,
    )
    write_json(scf / "neutral_scf.json", neutral)
    ne = int(case["active_space"]["electrons"])
    k = int(case["active_space"]["orbitals"])
    if ne % 2 or case["states"]["spin_multiplicity"] != 1:
        raise ValueError("Cold-start selector currently supports closed-shell singlet seeds")
    frozen = [i for i, t in enumerate(inp.type_indices) if t == "f"]
    deleted = [i for i, t in enumerate(inp.type_indices) if t == "d"]
    active, inactive = frontier_partition(inp.occupations, mo.energies, inp.type_indices, ne, k)
    ni = (inp.electron_count - ne) // 2 - len(frozen)
    if len(inactive) != ni:
        raise ValueError("SCF initial inactive complement is inconsistent")
    tags = [
        "f"
        if i in frozen
        else "d"
        if i in deleted
        else "2"
        if i in active
        else "i"
        if i in inactive
        else "s"
        for i in range(mo.orbital_count)
    ]
    prepared = directory / "prepared_scf"
    prepared.mkdir(exist_ok=True)
    cbasis, ccheckpoint = prepared / "input.scf.h5", prepared / "input.ScfOrb"
    if not cbasis.exists():
        shutil.copy2(basis, cbasis)
        with h5py.File(cbasis, "r+") as h:
            h["MO_TYPEINDICES"][...] = np.asarray(tags, dtype=h["MO_TYPEINDICES"].dtype)
        ccheckpoint.write_text(backend._replace_type_indices(checkpoint.read_text(), tags))
        write_json(
            prepared / "preparation.json",
            dict(
                role="SCF_orbital_partition_only_no_correlated_observation",
                source_basis=backend.binding(basis),
                source_checkpoint=backend.binding(checkpoint),
                active=active,
                inactive=inactive,
                initial_selection_used_composition=False,
                initial_selector="neutral_energy_frontier_v1",
            ),
        )
    role_reference = dict(
        basis_file=backend.binding(basis), checkpoint=backend.binding(checkpoint), active=active
    )
    context = backend.context_from_files(
        case, cbasis, ccheckpoint, config, electrons=ne, role_reference=role_reference
    )
    backend.check_capabilities(context, "optimize", config["channels"], config)
    result = backend.execute(context, active, "optimize", directory / "initial_optimize", config)
    if not all(result.get(key) for key in ("valid", "converged", "state_valid")):
        raise RuntimeError(
            "Initial orbital optimization failed; its receipt and costs are preserved"
        )
    initialization = dict(
        role="new_physical_cold_stage",
        fresh_scf=scf_source is None,
        source_case=source_case or case,
        initial_selection_used_composition=False,
        initial_selector="neutral_energy_frontier_v1",
        matches_search_protocol=True,
        cpu_seconds=timing["cpu_seconds"] + result["cpu_seconds"],
        wall_seconds=timing["wall_seconds"] + result["wall_seconds"],
        historical_cpu_seconds=timing.get("historical_cpu_seconds"),
        historical_wall_seconds=timing.get("historical_wall_seconds"),
        historical_costs_unknown=bool(scf_source and timing.get("historical_cpu_seconds") is None),
        source_scf=backend.binding(scf / "neutral_scf.json"),
        protocol=context.payload["protocol"],
        electron_role_reference=role_reference,
    )
    seed = dict(context=result["output_context"], incumbent=result, initialization=initialization)
    write_json(cached, seed)
    return seed
