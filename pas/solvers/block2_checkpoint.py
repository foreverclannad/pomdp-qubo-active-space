# Adapted from the user-supplied, verified checkpoint extractor.
# Original source hashes and adaptations: docs/review/reused_sources.json.
"""Exact evidence and energy from a copied immutable Block2 0.5.3 MPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from pas.common import identity as stable_payload_hash
from pas.common import write_json as _write_json_atomic
from pas.solvers.orbital_entropy import CONSISTENCY_TOLERANCE, orbital_entropies

REQUEST_SCHEMA = "pas_block2_checkpoint_request_v2"
RESULT_SCHEMA = "pas_block2_checkpoint_result_v2"


def _unavailable(
    request_id: str,
    reason: str,
    *,
    error_type: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "schema": RESULT_SCHEMA,
        "request_id": str(request_id),
        "status": "unavailable",
        "evidence_class": "unavailable",
        "reason": str(reason),
        "error_type": error_type,
        "error": error,
        "fallback_to_existing_proxy": False,
    }


def _read_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("schema") != REQUEST_SCHEMA:
        raise ValueError("invalid checkpoint entanglement request schema")
    unsigned = dict(payload)
    claimed = str(unsigned.pop("request_id", ""))
    if not claimed or stable_payload_hash(unsigned) != claimed:
        raise ValueError("checkpoint entanglement request hash mismatch")
    return dict(payload)


def _discover_work(source_directory: Path) -> Path:
    source = source_directory.expanduser().resolve()
    candidates = [source] if (source / "dmrg.conf").is_file() else []
    candidates.extend(path.parent for path in source.rglob("dmrg.conf"))
    valid = sorted(
        {
            path.resolve()
            for path in candidates
            if (path / "FCIDUMP").is_file()
            and (path / "dmrg.out").is_file()
            and (path / "node0").is_dir()
            and (path / "nodex").is_dir()
        }
    )
    if len(valid) != 1:
        raise RuntimeError(
            f"expected one final OpenMolcas Block2 work directory, found {len(valid)}"
        )
    return valid[0]


def _dmrg_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw in text.splitlines():
        fields = raw.split()
        if len(fields) >= 2:
            values.setdefault(fields[0].lower(), " ".join(fields[1:]))
    if "nonspinadapted" in text.lower() or "use_general_spin" in text.lower():
        raise RuntimeError("checkpoint is not an SU2 checkpoint")
    return values


def _fcidump_header(path: Path) -> dict[str, int]:
    header_lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            header_lines.append(line)
            if "/" in line:
                break
    header = " ".join(header_lines)
    result: dict[str, int] = {}
    for name in ("NORB", "NELEC", "MS2"):
        match = re.search(rf"\b{name}\s*=\s*(-?\d+)", header, re.IGNORECASE)
        if match is None:
            raise RuntimeError(f"FCIDUMP header has no {name}")
        result[name.lower()] = int(match.group(1))
    return result


def _orbital_reorder(path: Path, count: int) -> np.ndarray:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(
        r"(?:loading\s+reorder\s+for\s+restarting|reordering|using\s+gaopt\s+reorder)\s*=\s*\[([^\]]+)\]",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if not matches:
        raise RuntimeError("Block2 orbital reorder is unavailable")
    reorder = np.fromstring(matches[-1], dtype=int, sep=" ")
    saved = path.parent / "node0" / "orbital_reorder.npy"
    if saved.exists() and not np.array_equal(reorder, np.load(saved)):
        raise RuntimeError("Block2 log and checkpoint orbital orders differ")
    if reorder.size != count or sorted(reorder.tolist()) != list(range(count)):
        raise RuntimeError("Block2 orbital reorder is not a permutation")
    inverse = np.argsort(reorder)
    if not np.array_equal(reorder[inverse], np.arange(count)):
        raise RuntimeError("Block2 orbital reorder inverse failed")
    return reorder


def _tree_identity(work: Path) -> dict[str, Any]:
    paths = [work / "dmrg.conf", work / "dmrg.out", work / "FCIDUMP"]
    paths.extend(sorted((work / "node0").rglob("*")))
    paths.extend(sorted((work / "nodex").rglob("*")))
    files = [path for path in paths if path.is_file()]
    aggregate = hashlib.sha256()
    total = 0
    for path in files:
        relative = path.relative_to(work).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        size = int(path.stat().st_size)
        total += size
        aggregate.update(relative.encode("utf-8") + b"\0")
        aggregate.update(str(size).encode("ascii") + b"\0")
        aggregate.update(digest.digest())
    return {
        "file_count": len(files),
        "size_bytes": total,
        "aggregate_sha256": aggregate.hexdigest(),
    }


def _validated_checkpoint(
    request: Mapping[str, Any],
) -> tuple[Path, np.ndarray, dict[str, Any]]:
    work = _discover_work(Path(str(request["source_directory"])))
    active = tuple(int(value) for value in request["active_orbital_indices"])
    config = _dmrg_config(work / "dmrg.conf")
    header = _fcidump_header(work / "FCIDUMP")
    expected = {
        "norb": len(active),
        "nelec": int(request["active_electrons"]),
        "spin": int(request["spin"]),
        "nroots": len(request["state_ids"]),
    }
    observed = {
        "norb": int(header["norb"]),
        "nelec": int(config.get("nelec", -1)),
        "spin": int(config.get("spin", -1)),
        "nroots": int(config.get("nroots", -1)),
    }
    if observed != expected:
        raise RuntimeError(
            f"checkpoint quantum numbers differ: expected {expected}, got {observed}"
        )
    if header["nelec"] != expected["nelec"] or header["ms2"] != expected["spin"]:
        raise RuntimeError("FCIDUMP electron count or spin differs from request")
    tag = str(dict(request["checkpoint"])["mps_tag"])
    if config.get("mps_tags", "KET").split() != [tag]:
        raise RuntimeError("Requested MPS tag differs from the saved configuration")
    weights = np.asarray(request["state_weights"], dtype=float)
    if (
        weights.shape != (expected["nroots"],)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
        or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0)
    ):
        raise RuntimeError("Invalid requested MPS root weights")
    configured = (
        np.array([float(x) for x in config["weights"].split()])
        if "weights" in config
        else np.full(expected["nroots"], 1.0 / expected["nroots"])
    )
    if configured.shape != weights.shape or not np.allclose(
        configured, weights, atol=1e-12, rtol=0
    ):
        raise RuntimeError("Checkpoint configuration root weights differ")
    tag_info = work / "node0" / f"{tag}-mps_info.bin"
    if not tag_info.is_file():
        raise RuntimeError(f"missing Block2 MPS tag {tag}")
    if tag_info.stat().st_size < 128 or not any((work / "nodex").glob(f"*{tag}*")):
        raise RuntimeError(f"damaged Block2 MPS tag {tag}")
    reorder = _orbital_reorder(work / "dmrg.out", expected["norb"])
    return (
        work,
        reorder,
        {
            "dmrg_config": {key: config.get(key) for key in ("sym", "nelec", "nroots", "spin")},
            "fcidump_header": header,
            "orbital_reorder": {
                "dmrg_site_to_openmolcas_active_position": reorder.tolist(),
                "openmolcas_active_position_to_dmrg_site": np.argsort(reorder).tolist(),
                "roundtrip_verified": True,
            },
        },
    )


def _copy_checkpoint(work: Path, destination: Path) -> None:
    shutil.copytree(work / "node0", destination, dirs_exist_ok=True)
    shutil.copytree(work / "nodex", destination, dirs_exist_ok=True)


def _restore_openmolcas_order(values: np.ndarray, reorder: np.ndarray) -> np.ndarray:
    inverse = np.argsort(reorder)
    if values.ndim == 1:
        restored = values[inverse]
        if not np.allclose(restored[reorder], values, atol=0.0, rtol=0.0):
            raise RuntimeError("orbital reorder vector roundtrip failed")
        return restored
    restored = values[np.ix_(inverse, inverse)]
    if not np.allclose(restored[np.ix_(reorder, reorder)], values, atol=0.0, rtol=0.0):
        raise RuntimeError("orbital reorder matrix roundtrip failed")
    return restored


def _state_row(
    driver: Any,
    ket: Any,
    *,
    state_id: str,
    root_index: int,
    weight: float,
    reorder: np.ndarray,
) -> dict[str, Any]:
    count = int(reorder.size)
    s1, one_audit = orbital_entropies(driver, ket, orb_type=1)
    s2, two_audit = orbital_entropies(driver, ket, orb_type=2)
    if s1.size != count or s2.shape != (count, count):
        raise RuntimeError("Block2 entropy arrays have incompatible shapes")
    if not np.all(np.isfinite(s1)) or not np.all(np.isfinite(s2)):
        raise RuntimeError("Block2 entropy arrays contain non-finite values")
    diagonal_error = float(np.max(abs(np.diag(s2) - s1)))
    symmetry_error = float(np.max(abs(s2 - s2.T)))
    if max(diagonal_error, symmetry_error) > CONSISTENCY_TOLERANCE:
        raise RuntimeError("Orbital entropies violate diagonal or exchange consistency")
    s1 = _restore_openmolcas_order(s1, reorder)
    s2 = _restore_openmolcas_order(s2, reorder)
    mutual_information = 0.5 * (s1[:, None] + s1[None, :] - s2)
    mutual_information = 0.5 * (mutual_information + mutual_information.T)
    np.fill_diagonal(mutual_information, 0.0)
    if not np.all(np.isfinite(mutual_information)):
        raise RuntimeError("mutual information contains non-finite values")
    if np.min(mutual_information) < -CONSISTENCY_TOLERANCE:
        raise RuntimeError("Orbital entropies violate mutual information positivity")
    return {
        "state_id": str(state_id),
        "root_index": int(root_index),
        "weight": float(weight),
        "single_orbital_entropy": s1.tolist(),
        "two_orbital_entropy": s2.tolist(),
        "mutual_information": mutual_information.tolist(),
        "entropy_numerics": dict(
            one_orbital=one_audit,
            two_orbital=two_audit,
            diagonal_max_error=diagonal_error,
            exchange_max_error=symmetry_error,
            consistency_absolute_tolerance=CONSISTENCY_TOLERANCE,
        ),
    }


def _normalize_mps(driver, ket, *, label):
    identity_mpo = driver.get_identity_mpo()
    norm = complex(driver.expectation(ket, identity_mpo, ket, iprint=0))
    if not np.isfinite(norm) or abs(norm.imag) > 1e-10 or norm.real <= 0:
        raise RuntimeError(f"Invalid {label} norm")
    ket.iscale(1 / np.sqrt(norm.real))
    normalized = complex(driver.expectation(ket, identity_mpo, ket, iprint=0))
    if not np.isfinite(normalized) or abs(normalized - 1) > 1e-9:
        raise RuntimeError(f"{label} normalization failed")
    return norm.real, normalized.real


def _prepare_su2_state(driver, ket, tag, *, electrons, spin):
    """One representation for density, energy and subsequent spin projection."""
    target = ket.info.target
    if (int(target.n), int(target.twos), int(target.pg)) != (electrons, spin, 0):
        raise RuntimeError("MPS target electron/spin differs")
    if ket.center != 0:
        driver.align_mps_center(ket, ref=0)
    if spin:
        # NSE density contractions in the pinned driver can lose electron trace.
        # Embed before *all* observables, not only before entropy projection.
        ket = driver.mps_change_to_singlet_embedding(ket, tag + "_SU2_EMBEDDED")
    target = ket.info.target
    if (int(target.n), int(target.twos), int(target.pg)) != (electrons + spin, 0, 0):
        raise RuntimeError("Invalid prepared SU2 MPS quantum numbers")
    norm, normalized = _normalize_mps(driver, ket, label="SU2 state")
    return ket, dict(
        representation="singlet_embedding" if spin else "singlet",
        electrons=electrons,
        twice_total_spin=spin,
        target_electrons=int(target.n),
        original_norm_squared=norm,
        normalized_norm_squared=normalized,
    )


def _convert_spin_projection(driver, ket, tag, spin):
    if int(ket.info.target.twos) != 0:
        raise RuntimeError("Spin projection requires a prepared singlet-embedded SU2 state")
    if ket.center != 0:
        driver.align_mps_center(ket, ref=0)
    return driver.mps_change_to_sz(ket, tag, sz=spin)


def _normalize_spin_projection(driver, ket, *, electrons, spin):
    target = ket.info.target
    if (int(target.n), int(target.twos), int(target.pg)) != (electrons, spin, 0):
        raise RuntimeError("Invalid projected MPS quantum numbers")
    norm, normalized = _normalize_mps(driver, ket, label="spin projection")
    return dict(
        projection_norm_squared=norm,
        normalized_norm_squared=normalized,
        electrons=electrons,
        twice_spin_projection=spin,
        point_group_irrep=0,
    )


def extract_checkpoint_entanglement(request, *, temporary_directory):
    """Read exactly the saved states: diagonal 1-RDM, optional QIT, and <H>.

    There are no transition-state operators, continuity model, approximate
    entanglement, or density files emitted by this adapter. Energy and density
    come from each same split final MPS, in the bound original FCIDUMP basis.
    """
    if list(request.get("extraction_strategies", ())) != ["checkpoint_mps"]:
        raise ValueError("Only actual checkpoint MPS extraction is supported")
    checkpoint = dict(request["checkpoint"])
    if checkpoint.get("symmetry") != "SU2":
        raise ValueError("Only SU2 is supported")
    requested = set(request["requested_evidence"])
    if not requested or not requested <= {
        "occupation",
        "single_orbital_entropy",
        "mutual_information",
    }:
        raise ValueError("Unsupported evidence request")
    qit = bool(requested & {"single_orbital_entropy", "mutual_information"})
    work, reorder, audit = _validated_checkpoint(request)
    before = _tree_identity(work)
    temporary_directory = Path(temporary_directory)
    temporary_directory.mkdir(parents=True, exist_ok=True)
    _copy_checkpoint(work, temporary_directory)
    from pyblock2.driver.core import DMRGDriver, SymmetryTypes

    driver = DMRGDriver(
        scratch=str(temporary_directory),
        clean_scratch=False,
        symm_type=SymmetryTypes.SU2,
        n_threads=int(checkpoint["threads"]),
        stack_mem=int(checkpoint["stack_memory_bytes"]),
    )
    count = len(reorder)
    driver.initialize_system(
        n_sites=count,
        n_elec=int(request["active_electrons"]),
        spin=int(request["spin"]),
        orb_sym=[0] * count,
    )
    nroots = len(request["state_ids"])
    multi = driver.load_mps(checkpoint["mps_tag"], nroots=nroots)
    if nroots > 1:
        weights = np.asarray(multi.weights)
        if weights.shape != (nroots,) or not np.allclose(
            weights, request["state_weights"], atol=1e-12, rtol=0
        ):
            raise RuntimeError("MPS root weights differ")
    roots = []
    densities = []
    states = []
    preparation_audit = []
    for i, (state, weight) in enumerate(
        zip(request["state_ids"], request["state_weights"], strict=True)
    ):
        # A native single-root checkpoint is an ordinary MPS: it has neither
        # state-average weights nor a MultiMPS wavefunction to split.
        ket = multi if nroots == 1 else driver.split_mps(multi, i, f"PAS_ROOT_{i}")
        ket, prepared = _prepare_su2_state(
            driver,
            ket,
            f"PAS_ROOT_{i}",
            electrons=int(request["active_electrons"]),
            spin=int(request["spin"]),
        )
        roots.append(ket)
        raw = np.asarray(driver.get_1pdm(ket, iprint=0))
        if (
            raw.shape != (count, count)
            or not np.all(np.isfinite(raw))
            or np.max(abs(raw.imag)) > 1e-10
        ):
            raise RuntimeError("Invalid real complete MPS density")
        trace = float(np.trace(raw).real)
        if abs(trace - request["active_electrons"]) > 1e-6:
            raise RuntimeError("Prepared SU2 density electron trace differs")
        prepared.update(
            state_id=state, electron_trace=trace, electron_trace_absolute_tolerance=1e-6
        )
        preparation_audit.append(prepared)
        densities.append(_restore_openmolcas_order(raw.real, reorder).tolist())
    # Build H with the exact saved permutation only after extracting densities:
    # get_qc_mpo sets driver.reorder_idx, which would otherwise implicitly
    # unpermute subsequent NPDMs a second time.
    driver.read_fcidump(str(work / "FCIDUMP"), pg="c1", iprint=0)
    mpo = driver.get_qc_mpo(driver.h1e, driver.g2e, ecore=driver.ecore, reorder=reorder, iprint=0)
    energies = np.asarray([driver.expectation(ket, mpo, ket, iprint=0) for ket in roots], float)
    if not np.all(np.isfinite(energies)):
        raise RuntimeError("Nonfinite final MPS expectation energy")
    extraction_mode = "su2_checkpoint_1rdm_energy"
    projection_audit = []
    if qit:
        # Block2 0.5.3's SU2 entropy NPDM interface does not accept every
        # coupled operator mask. Convert exactly to the fixed spin projection;
        # no sweeps, truncation, alternative solver or proxy are involved.
        driver.reorder_idx = None
        tags = []
        for i, ket in enumerate(roots):
            tag = f"PAS_SZ_ROOT_{i}"
            projected = _convert_spin_projection(driver, ket, tag, int(request["spin"]))
            tags.append(tag)
        del mpo, roots, multi, ket, projected
        import gc

        gc.collect()
        driver.set_symm_type(SymmetryTypes.SZ)
        driver.initialize_system(
            n_sites=count,
            n_elec=int(request["active_electrons"]),
            spin=int(request["spin"]),
            orb_sym=[0] * count,
        )
        for i, tag in enumerate(tags):
            ket = driver.load_mps(tag)
            audit_row = _normalize_spin_projection(
                driver, ket, electrons=int(request["active_electrons"]), spin=int(request["spin"])
            )
            projected_density = np.asarray(driver.get_1pdm(ket, iprint=0))
            if projected_density.shape != (2, count, count) or not np.all(
                np.isfinite(projected_density)
            ):
                raise RuntimeError("Invalid normalized spin-projection density")
            restored = _restore_openmolcas_order(projected_density.sum(axis=0), reorder)
            error = float(np.max(abs(restored - np.asarray(densities[i]))))
            if error > 1e-8:
                raise RuntimeError("Spin projection changed the spin-summed MPS density")
            audit_row.update(
                state_id=request["state_ids"][i],
                density_max_error=error,
                density_absolute_tolerance=1e-8,
            )
            projection_audit.append(audit_row)
            _write_json_atomic(temporary_directory / "spin_projection.json", projection_audit)
            states.append(
                _state_row(
                    driver,
                    ket,
                    state_id=request["state_ids"][i],
                    root_index=i,
                    weight=request["state_weights"][i],
                    reorder=reorder,
                )
            )
        extraction_mode = "exact_su2_to_sz_checkpoint_1rdm_energy_qit"
    sweeps = np.asarray(np.load(work / "node0/sweep_energies.npy"), float)
    if sweeps.ndim != 2 or sweeps.shape[1] != len(energies) or not np.all(np.isfinite(sweeps)):
        raise RuntimeError("Invalid final sweep energy binding")
    gaps = energies - sweeps[-1]
    # The sweep energy can precede truncation. Use the actual final MPS <H>
    # paired with that MPS's density, recording the discrepancy explicitly.
    # No finite-M error bound is inferred from the Davidson/sweep tolerance.
    if _tree_identity(work) != before:
        raise RuntimeError("Source checkpoint changed during extraction")
    result = dict(
        schema=RESULT_SCHEMA,
        request_id=request["request_id"],
        status="available",
        orbital_basis_id=request["orbital_basis_id"],
        physical_evidence_id=request["physical_evidence_id"],
        evidence_class="true_entanglement" if qit else "state_density",
        provider="block2",
        extraction_mode=extraction_mode,
        active_orbital_indices=list(request["active_orbital_indices"]),
        state_ids=list(request["state_ids"]),
        state_weights=list(request["state_weights"]),
        su2_state_preparation=preparation_audit,
        state_operator_order=[dict(bra_root=i, ket_root=i) for i in range(len(energies))],
        state_operators=densities,
        root_energies=energies.tolist(),
        energy_source="expectation_of_same_final_MPS_as_density",
        sweep_root_energies=sweeps[-1].tolist(),
        energy_density_gaps=gaps.tolist(),
        density_energy_endpoint="same_final_MPS",
        finite_m_error_certified=False,
        states=states,
        checkpoint=dict(
            work_directory=str(work),
            mps_tag=checkpoint["mps_tag"],
            symmetry="SU2",
            identity=before,
            **audit,
        ),
        source_checkpoint_unchanged=True,
        dmrg_sweeps_executed=0,
        fallback_to_existing_proxy=False,
    )
    if qit:
        result.update(
            entropy_unit="natural_log",
            mutual_information_convention="0.5*(s_i+s_j-s_ij)",
            spin_projection=projection_audit,
        )
    result["result_id"] = stable_payload_hash(result)
    return result


def cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--temporary-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    request_id = ""
    try:
        request = _read_request(args.request)
        request_id = str(request["request_id"])
        result = extract_checkpoint_entanglement(
            request,
            temporary_directory=args.temporary_directory,
        )
        _write_json_atomic(args.result, result)
        return 0
    except Exception as exc:
        traceback.print_exc()
        _write_json_atomic(
            args.result,
            _unavailable(
                request_id,
                "checkpoint_extraction_failed",
                error_type=type(exc).__name__,
                error=str(exc),
            ),
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(cli_main())
