"""Independent reference acceptance; never an input to search or Bayesian inference.

The numerical comparators below are vendored from minimal-valence-control-suite,
post_stop_comparison.py, SHA256 0286d1955a2cedf9b20a2d58c3602b1fb7c8458179815d4eb0a9a47d6cf081c5.
The physical overlap/SVD algorithms remain independently audited. Shared physical
INPORB parsing lives in orbital_io; no old repository is a runtime dependency.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import file_hash
from .orbital_io import read_openmolcas_orbital_hdf5, read_rasorb_coefficients

COMPARISON_ROLE = "external_development_acceptance"
REPORT_SCHEMA = "openmolcas_post_stop_active_subspace_comparison_v2"
ACTIVE_TYPES = frozenset({"1", "2", "3"})
HDF5_COMPARATOR_ID = "same_ao_basis_hdf5_subspace_v1"
RASORB_COMPARATOR_ID = "hdf5_to_inporb_same_ao_basis_v1"
SUPPORTED_COMPARATORS = frozenset({HDF5_COMPARATOR_ID, RASORB_COMPARATOR_ID})
BOHR_PER_ANGSTROM = 1.0 / 0.529177210903


class PostStopComparisonError(RuntimeError):
    """Raised when a held-out comparison cannot be made safely."""


def sha256_file(path: str | Path) -> str:
    """Return the SHA256 of one required regular file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return file_hash(source)


def comparator_code_sha256() -> str:
    """Hash the implementation that defines all comparator semantics."""

    return sha256_file(Path(__file__).resolve())


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _comparator_record(comparator_id: str) -> dict[str, Any]:
    if comparator_id not in SUPPORTED_COMPARATORS:
        raise PostStopComparisonError(f"unsupported held-out comparator: {comparator_id!r}")
    return {
        "id": comparator_id,
        "implementation": "pas.acceptance",
        "code_sha256": comparator_code_sha256(),
        "orbital_io_sha256": sha256_file(Path(__file__).with_name("orbital_io.py")),
    }


def _file_record(path: Path) -> dict[str, str]:
    resolved = path.expanduser().resolve()
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def _active_indices(
    type_indices: Sequence[str],
    *,
    explicit: Sequence[int] | None,
    expected_count: int | None,
    role: str,
) -> tuple[int, ...]:
    orbital_count = len(type_indices)
    if explicit is None:
        result = tuple(
            index
            for index, value in enumerate(type_indices)
            if str(value).strip().lower() in ACTIVE_TYPES
        )
    else:
        result = tuple(int(value) for value in explicit)
    if not result:
        raise PostStopComparisonError(f"{role} active-space indices are empty")
    if len(set(result)) != len(result):
        raise PostStopComparisonError(f"{role} active-space indices are not unique")
    if min(result) < 0 or max(result) >= orbital_count:
        raise PostStopComparisonError(f"{role} active-space index is outside [0, {orbital_count})")
    if expected_count is not None and len(result) != int(expected_count):
        raise PostStopComparisonError(
            f"{role} active-space dimension {len(result)} != {int(expected_count)}"
        )
    return result


def _orthonormality_residual(
    coefficients: np.ndarray,
    overlap: np.ndarray,
) -> float:
    coefficients = np.asarray(coefficients, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    if coefficients.ndim != 2 or overlap.shape != (
        coefficients.shape[0],
        coefficients.shape[0],
    ):
        raise PostStopComparisonError("coefficient/metric dimensions are inconsistent")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(overlap)):
        raise PostStopComparisonError("coefficient/metric arrays contain non-finite values")
    residual = coefficients.T @ overlap @ coefficients - np.eye(coefficients.shape[1], dtype=float)
    return float(np.max(np.abs(residual))) if residual.size else 0.0


def _metric_orthonormalize(
    coefficients: np.ndarray,
    overlap: np.ndarray,
    *,
    role: str,
    rcond: float,
) -> tuple[np.ndarray, np.ndarray]:
    coefficients = np.asarray(coefficients, dtype=float)
    overlap = np.asarray(overlap, dtype=float)
    if coefficients.ndim != 2 or coefficients.shape[1] == 0:
        raise PostStopComparisonError(f"{role} coefficient block is empty or invalid")
    if overlap.shape != (coefficients.shape[0], coefficients.shape[0]):
        raise PostStopComparisonError(f"{role} AO metric has an incompatible shape")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(overlap)):
        raise PostStopComparisonError(f"{role} coefficients/metric are non-finite")
    gram = coefficients.T @ overlap @ coefficients
    gram = 0.5 * (gram + gram.T)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    threshold = max(float(rcond), 0.0) * scale
    if float(np.min(eigenvalues)) <= threshold:
        raise PostStopComparisonError(
            f"{role} active subspace is singular in its AO metric: "
            f"minimum_eigenvalue={float(np.min(eigenvalues)):.6e}"
        )
    inverse_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return coefficients @ inverse_sqrt, eigenvalues


def _subspace_payload(
    selected_coefficients: np.ndarray,
    reference_coefficients: np.ndarray,
    *,
    selected_metric: np.ndarray,
    reference_metric: np.ndarray,
    cross_metric: np.ndarray,
    rcond: float,
    singular_value_tolerance: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    selected_coefficients = np.asarray(selected_coefficients, dtype=float)
    reference_coefficients = np.asarray(reference_coefficients, dtype=float)
    if selected_coefficients.shape[1] != reference_coefficients.shape[1]:
        raise PostStopComparisonError("selected/reference active spaces have different dimensions")
    if np.asarray(cross_metric).shape != (
        selected_coefficients.shape[0],
        reference_coefficients.shape[0],
    ):
        raise PostStopComparisonError("cross AO overlap has an incompatible shape")
    selected_orth, selected_eigenvalues = _metric_orthonormalize(
        selected_coefficients,
        selected_metric,
        role="selected",
        rcond=rcond,
    )
    reference_orth, reference_eigenvalues = _metric_orthonormalize(
        reference_coefficients,
        reference_metric,
        role="reference",
        rcond=rcond,
    )
    principal_overlap = selected_orth.T @ cross_metric @ reference_orth
    raw_singular_values = np.linalg.svd(principal_overlap, compute_uv=False)
    if not np.all(np.isfinite(raw_singular_values)):
        raise PostStopComparisonError("principal singular values are non-finite")
    maximum = float(np.max(raw_singular_values)) if raw_singular_values.size else 0.0
    if maximum > 1.0 + float(singular_value_tolerance):
        raise PostStopComparisonError(
            f"principal singular value exceeds unity beyond numerical tolerance: {maximum:.12g}"
        )
    singular_values = np.clip(raw_singular_values, 0.0, 1.0)
    angles = np.arccos(singular_values)
    degrees = np.degrees(angles)
    dimension = int(singular_values.size)
    subspace = {
        "available": True,
        "dimension": dimension,
        "singular_values": [float(value) for value in singular_values],
        "minimum_singular_value": (float(np.min(singular_values)) if dimension else 1.0),
        "mean_singular_value": (float(np.mean(singular_values)) if dimension else 1.0),
        "mean_squared_singular_value": (float(np.mean(singular_values**2)) if dimension else 1.0),
        "sum_squared_singular_values": float(np.sum(singular_values**2)),
        "principal_angles_radians": [float(value) for value in angles],
        "principal_angles_degrees": [float(value) for value in degrees],
        "maximum_principal_angle_degrees": (float(np.max(degrees)) if dimension else 0.0),
        "directions_at_or_above_0_90": int(np.count_nonzero(singular_values >= 0.90)),
        "directions_at_or_above_0_95": int(np.count_nonzero(singular_values >= 0.95)),
    }
    normalization = {
        "subspace_normalization_model": (
            "symmetric_metric_orthonormalized_principal_singular_values_v1"
        ),
        "selected_active_gram_eigenvalue_range_before_normalization": [
            float(np.min(selected_eigenvalues)),
            float(np.max(selected_eigenvalues)),
        ],
        "reference_active_gram_eigenvalue_range_before_normalization": [
            float(np.min(reference_eigenvalues)),
            float(np.max(reference_eigenvalues)),
        ],
        "selected_active_orthonormality_max_abs_residual_before_normalization": (
            _orthonormality_residual(selected_coefficients, selected_metric)
        ),
        "reference_active_orthonormality_max_abs_residual_before_normalization": (
            _orthonormality_residual(reference_coefficients, reference_metric)
        ),
    }
    return subspace, normalization


def _hdf5_auxiliary(path: Path) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise PostStopComparisonError("HDF5 comparison requires h5py") from exc
    with h5py.File(path, mode="r") as handle:
        roots = (
            np.asarray(handle["ROOT_ENERGIES"][()], dtype=float).reshape(-1)
            if "ROOT_ENERGIES" in handle
            else np.asarray([], dtype=float)
        )
        coordinate_key = (
            "DESYM_CENTER_COORDINATES"
            if "DESYM_CENTER_COORDINATES" in handle
            else "CENTER_COORDINATES"
        )
        coordinates = (
            np.asarray(handle[coordinate_key][()], dtype=float)
            if coordinate_key in handle
            else None
        )
    if roots.size and not np.all(np.isfinite(roots)):
        raise PostStopComparisonError("HDF5 root energies are non-finite")
    return {
        "root_energies_hartree": [float(value) for value in roots],
        "center_coordinates_bohr": coordinates,
    }


def _base_report(
    *,
    comparator_id: str,
    label: str,
    selected: Mapping[str, Any],
    reference: Mapping[str, Any],
    basis_compatibility: Mapping[str, Any],
    subspace: Mapping[str, Any],
) -> dict[str, Any]:
    comparator = _comparator_record(comparator_id)
    return {
        "schema": REPORT_SCHEMA,
        "status": "complete",
        "available": True,
        "comparison_role": COMPARISON_ROLE,
        "label": str(label),
        "generated_at_utc": _timestamp(),
        "comparator": comparator,
        "comparator_id": comparator["id"],
        "comparator_code_sha256": comparator["code_sha256"],
        "selected": dict(selected),
        "reference": dict(reference),
        "basis_compatibility": dict(basis_compatibility),
        "subspace": dict(subspace),
        "fresh_binding": {
            "candidate_sha256": selected.get("sha256"),
            "reference_sha256": reference.get("sha256"),
            "subspace_dimension": subspace.get("dimension"),
            "comparator_id": comparator["id"],
            "comparator_code_sha256": comparator["code_sha256"],
        },
    }


def compare_hdf5_active_subspaces(
    selected_path: str | Path,
    reference_path: str | Path,
    *,
    expected_active_orbitals: int | None = None,
    selected_active_indices: Sequence[int] | None = None,
    reference_active_indices: Sequence[int] | None = None,
    ao_overlap_tolerance: float = 1.0e-8,
    metric_rcond: float = 1.0e-10,
    singular_value_tolerance: float = 1.0e-6,
    label: str = "",
) -> dict[str, Any]:
    """Compare two HDF5 active spaces after strict same-AO-basis checks."""

    selected_source = Path(selected_path).expanduser().resolve()
    reference_source = Path(reference_path).expanduser().resolve()
    selected = read_openmolcas_orbital_hdf5(selected_source)
    reference = read_openmolcas_orbital_hdf5(reference_source)
    selected_aux = _hdf5_auxiliary(selected_source)
    reference_aux = _hdf5_auxiliary(reference_source)
    if selected.orbital_count != reference.orbital_count:
        raise PostStopComparisonError("HDF5 orbital counts differ")
    if selected.center_labels != reference.center_labels:
        raise PostStopComparisonError("HDF5 center labels differ")
    if not np.array_equal(selected.basis_function_ids, reference.basis_function_ids):
        raise PostStopComparisonError("HDF5 AO basis-function identities differ")
    overlap_delta = float(np.max(np.abs(selected.ao_overlap - reference.ao_overlap)))
    if overlap_delta > float(ao_overlap_tolerance):
        raise PostStopComparisonError(
            "HDF5 AO overlap matrices differ beyond tolerance: "
            f"{overlap_delta:.6e} > {float(ao_overlap_tolerance):.6e}"
        )
    selected_indices = _active_indices(
        selected.type_indices,
        explicit=selected_active_indices,
        expected_count=expected_active_orbitals,
        role="selected",
    )
    reference_indices = _active_indices(
        reference.type_indices,
        explicit=reference_active_indices,
        expected_count=expected_active_orbitals,
        role="reference",
    )
    subspace, normalization = _subspace_payload(
        selected.mo_coefficients[:, selected_indices],
        reference.mo_coefficients[:, reference_indices],
        selected_metric=selected.ao_overlap,
        reference_metric=reference.ao_overlap,
        cross_metric=selected.ao_overlap,
        rcond=float(metric_rcond),
        singular_value_tolerance=float(singular_value_tolerance),
    )
    selected_coordinates = selected_aux.pop("center_coordinates_bohr")
    reference_coordinates = reference_aux.pop("center_coordinates_bohr")
    coordinate_delta = None
    if selected_coordinates is not None or reference_coordinates is not None:
        if selected_coordinates is None or reference_coordinates is None:
            raise PostStopComparisonError("only one HDF5 file contains center coordinates")
        if selected_coordinates.shape != reference_coordinates.shape:
            raise PostStopComparisonError("HDF5 center-coordinate shapes differ")
        coordinate_delta = float(np.max(np.abs(selected_coordinates - reference_coordinates)))
    selected_record = {
        **_file_record(selected_source),
        "active_indices_zero_based": list(selected_indices),
        "active_indices_one_based": [index + 1 for index in selected_indices],
        **selected_aux,
    }
    reference_record = {
        **_file_record(reference_source),
        "active_indices_zero_based": list(reference_indices),
        "active_indices_one_based": [index + 1 for index in reference_indices],
        **reference_aux,
    }
    return _base_report(
        comparator_id=HDF5_COMPARATOR_ID,
        label=label,
        selected=selected_record,
        reference=reference_record,
        basis_compatibility={
            "same_ao_basis_verified": True,
            "orbital_count": selected.orbital_count,
            "symmetry_block_sizes": list(selected.symmetry_block_sizes),
            "center_labels_equal": True,
            "basis_function_ids_equal": True,
            "ao_overlap_max_abs_delta": overlap_delta,
            "ao_overlap_compatibility_tolerance": float(ao_overlap_tolerance),
            "center_coordinates_max_abs_delta_bohr": coordinate_delta,
            "selected_mo_orthonormality_max_abs_residual": (
                _orthonormality_residual(selected.mo_coefficients, selected.ao_overlap)
            ),
            "reference_mo_orthonormality_max_abs_residual": (
                _orthonormality_residual(reference.mo_coefficients, reference.ao_overlap)
            ),
            **normalization,
        },
        subspace=subspace,
    )


def compare_hdf5_to_rasorb(
    selected_path: str | Path,
    reference_path: str | Path,
    *,
    expected_active_orbitals: int | None = None,
    selected_active_indices: Sequence[int] | None = None,
    reference_active_indices: Sequence[int] | None = None,
    metric_rcond: float = 1.0e-10,
    singular_value_tolerance: float = 1.0e-6,
    full_mo_orthonormality_tolerance: float = 1.0e-3,
    label: str = "",
) -> dict[str, Any]:
    """Compare candidate HDF5 with a same-AO-basis INPORB/RasOrb."""

    selected_source = Path(selected_path).expanduser().resolve()
    reference_source = Path(reference_path).expanduser().resolve()
    selected = read_openmolcas_orbital_hdf5(selected_source)
    selected_aux = _hdf5_auxiliary(selected_source)
    selected_aux.pop("center_coordinates_bohr", None)
    reference_coefficients, reference_types = read_rasorb_coefficients(
        reference_source,
        symmetry_block_sizes=selected.symmetry_block_sizes,
    )
    reference_coefficients = selected.native_coefficients_to_ao(reference_coefficients)
    selected_indices = _active_indices(
        selected.type_indices,
        explicit=selected_active_indices,
        expected_count=expected_active_orbitals,
        role="selected",
    )
    reference_indices = _active_indices(
        reference_types,
        explicit=reference_active_indices,
        expected_count=expected_active_orbitals,
        role="reference",
    )
    selected_full_residual = _orthonormality_residual(selected.mo_coefficients, selected.ao_overlap)
    reference_full_residual = _orthonormality_residual(reference_coefficients, selected.ao_overlap)
    if selected_full_residual > float(full_mo_orthonormality_tolerance):
        raise PostStopComparisonError(
            "candidate HDF5 orbitals are not orthonormal in their AO metric"
        )
    if reference_full_residual > float(full_mo_orthonormality_tolerance):
        raise PostStopComparisonError(
            "reference INPORB is not compatible with the candidate AO metric: "
            f"residual={reference_full_residual:.6e}"
        )
    subspace, normalization = _subspace_payload(
        selected.mo_coefficients[:, selected_indices],
        reference_coefficients[:, reference_indices],
        selected_metric=selected.ao_overlap,
        reference_metric=selected.ao_overlap,
        cross_metric=selected.ao_overlap,
        rcond=float(metric_rcond),
        singular_value_tolerance=float(singular_value_tolerance),
    )
    selected_record = {
        **_file_record(selected_source),
        "active_indices_zero_based": list(selected_indices),
        "active_indices_one_based": [index + 1 for index in selected_indices],
        **selected_aux,
    }
    reference_record = {
        **_file_record(reference_source),
        "source_orbital_sha256": sha256_file(reference_source),
        "active_indices_zero_based": list(reference_indices),
        "active_indices_one_based": [index + 1 for index in reference_indices],
    }
    return _base_report(
        comparator_id=RASORB_COMPARATOR_ID,
        label=label,
        selected=selected_record,
        reference=reference_record,
        basis_compatibility={
            "same_ao_basis_verified": True,
            "verification_model": (
                "candidate_hdf5_block_layout_and_reference_full_mo_orthonormality_"
                "in_candidate_ao_metric_v1"
            ),
            "orbital_count": selected.orbital_count,
            "symmetry_block_sizes": list(selected.symmetry_block_sizes),
            "selected_mo_orthonormality_max_abs_residual": selected_full_residual,
            "reference_mo_orthonormality_max_abs_residual": reference_full_residual,
            "full_mo_orthonormality_tolerance": float(full_mo_orthonormality_tolerance),
            **normalization,
        },
        subspace=subspace,
    )


def _read_xyz(path: Path) -> tuple[tuple[str, ...], np.ndarray]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise PostStopComparisonError("XYZ file is empty")
    try:
        atom_count = int(lines[0].strip())
    except ValueError as exc:
        raise PostStopComparisonError("XYZ atom count is invalid") from exc
    rows = []
    labels = []
    for line in lines[2 : 2 + atom_count]:
        pieces = line.split()
        if len(pieces) < 4:
            raise PostStopComparisonError("XYZ atom row is incomplete")
        labels.append(str(pieces[0]))
        rows.append([float(value) for value in pieces[1:4]])
    if len(rows) != atom_count:
        raise PostStopComparisonError("XYZ atom count does not match its rows")
    coordinates = np.asarray(rows, dtype=float) * BOHR_PER_ANGSTROM
    if not np.all(np.isfinite(coordinates)):
        raise PostStopComparisonError("XYZ coordinates contain non-finite values")
    return tuple(labels), coordinates


def _checked_file(binding):
    path = Path(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        raise ValueError("Artifact hash changed: " + str(path))
    return path


def _case_name(incumbent):
    context = incumbent.get("output_context") or incumbent.get("context") or {}
    case = context.get("payload", {}).get("case", {})
    return incumbent.get("case_id") or case.get("key") or case.get("case_id")


def _physical_state(path):
    import h5py

    with h5py.File(path) as handle:
        attrs = handle.attrs
        if (
            attrs.get("SPINMULT") != 1
            or attrs.get("NACTEL") != 16
            or attrs.get("NROOTS") != 2
            or attrs.get("NSTATES") != 2
            or list(attrs.get("STATE_ROOTID", [])) != [1, 2]
            or np.shape(attrs.get("STATE_WEIGHT", [])) != (2,)
            or not np.allclose(attrs["STATE_WEIGHT"], [0.5, 0.5], atol=1e-12, rtol=0)
        ):
            raise ValueError("Incumbent is not the required SA2 singlet CAS(16e,13o) state")
        energies = np.asarray(handle["ROOT_ENERGIES"][:], dtype=float)
        if energies.shape != (2,) or not np.isfinite(energies).all():
            raise ValueError("Missing or invalid incumbent state energies")


def _check_geometry(path, geometry):
    from .orbital_io import _element_from_center_label

    symbols, xyz = _read_xyz(geometry)
    data = read_openmolcas_orbital_hdf5(path)
    if tuple(_element_from_center_label(s) for s in data.center_labels) != symbols:
        raise ValueError("Incumbent and bound reference geometry have different atoms")
    coordinates = _hdf5_auxiliary(path)["center_coordinates_bohr"]
    if (
        coordinates is None
        or coordinates.shape != xyz.shape
        or not np.isfinite(coordinates).all()
        or np.max(abs(coordinates - xyz)) > 2e-6
    ):
        raise ValueError("Incumbent and bound reference geometry differ")


def _compare_ci_oo(selected, references):
    import h5py

    reference = _checked_file(references["reference"])
    with h5py.File(selected) as left, h5py.File(reference) as right:
        a, b = left["CENTER_COORDINATES"][:], right["CENTER_COORDINATES"][:]
        if a.shape != b.shape or np.max(abs(a - b)) > 1e-8:
            raise ValueError("CI-OO reference coordinates differ from the calculation")
    return {
        "reference_subspace": compare_hdf5_active_subspaces(
            selected, reference, expected_active_orbitals=13
        )
    }


def _spectrum_passes(comparison):
    subspace = comparison["subspace"]
    values = np.asarray(subspace.get("singular_values", []), dtype=float)
    if values.shape != (13,) or not np.isfinite(values).all() or subspace.get("dimension") != 13:
        raise ValueError("Acceptance requires exactly 13 finite principal singular values")
    if np.any(values < 0) or np.any(values > 1):
        raise ValueError("Principal singular values outside [0,1]")
    count = int(np.count_nonzero(values >= 0.95))
    if subspace.get("directions_at_or_above_0_95") != count:
        raise ValueError("Coverage count differs from the complete singular spectrum")
    return count == 13


def audit(incumbent, references, output=None):
    """Audit one formal incumbent; malformed provenance raises, coverage miss returns false."""
    from .common import identity, read_json, write_json

    if isinstance(references, (str, Path)):
        references = read_json(references)
    if not all(incumbent.get(k) is True for k in ("valid", "converged", "state_valid")):
        raise ValueError("Only a completed physically valid incumbent can be audited")
    if incumbent.get("kind") not in ("initial", "optimize"):
        raise ValueError("Probe, survey, pending and unaccepted visit are not optimized incumbents")
    if incumbent.get("accepted") is False:
        raise ValueError("A rejected candidate cannot be audited as the formal incumbent")
    if not incumbent.get("evidence_id"):
        raise ValueError("Incumbent evidence identity is missing")
    case = _case_name(incumbent)
    if case not in ("NX1", "CI_OO_N00", "CI_CC", "TS_CC_P1"):
        raise ValueError("Unknown four-point case")
    if references.get("case_id", case) != case:
        raise ValueError("Reference binding belongs to another case")
    selected = _checked_file(incumbent["artifacts"]["basis_file"])
    if "checkpoint" in incumbent["artifacts"]:
        _checked_file(incumbent["artifacts"]["checkpoint"])
    _physical_state(selected)
    if case == "CI_OO_N00":
        comparisons = _compare_ci_oo(selected, references)
        required = {"reference_subspace"}
    else:
        mode = references.get("comparison_mode")
        if mode not in ("same_geometry_hdf5", "same_geometry_rasorb"):
            raise ValueError("Unsupported same-geometry reference binding")
        reference = _checked_file(references["reference"])
        geometry = _checked_file(references["reference_geometry"])
        _check_geometry(selected, geometry)
        if mode == "same_geometry_hdf5":
            _check_geometry(reference, geometry)
        catalog = json.loads(_checked_file(references["provenance_catalog"]).read_text())
        record = next((r for r in catalog["cases"] if r["structure"] == case), None)
        if (
            not record
            or record["heldout_baseline"]["reference_sha256"] != references["reference"]["sha256"]
        ):
            raise ValueError("Reference is not the archived baseline for this case")
        compare = (
            compare_hdf5_active_subspaces
            if mode == "same_geometry_hdf5"
            else compare_hdf5_to_rasorb
        )
        comparisons = {
            "reference_subspace": compare(selected, reference, expected_active_orbitals=13)
        }
        required = {"reference_subspace"}
    if set(comparisons) != required:
        raise ValueError("Required reference comparisons missing")
    passed = all([_spectrum_passes(comparisons[name]) for name in sorted(required)])
    _checked_file(incumbent["artifacts"]["basis_file"])
    report = dict(
        schema="pas_fourpoint_acceptance_v1",
        passed=passed,
        case_id=case,
        incumbent_evidence_id=incumbent["evidence_id"],
        incumbent_sha256=identity(incumbent),
        required_directions=13,
        singular_value_threshold=0.95,
        comparisons=comparisons,
        reference_binding_id=identity(references),
        comparator_sha256=comparator_code_sha256(),
        checked_utc=_timestamp(),
        reference_data_fed_to_search=False,
        physical_jobs_launched=0,
        interpretation="development_reference_coverage_not_optimality_or_method_superiority",
    )
    if output is not None:
        write_json(output, report)
    return report
