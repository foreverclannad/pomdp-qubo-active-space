"""Minimal, system-agnostic AVAS -> POMDP -> fixed-budget QUBO selector.

This module intentionally contains no molecule catalogue and no chemistry
special cases.  Molecules, AO targets, budgets, and post-run expectations live
in JSON inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CaseSpec:
    """One fixed-budget active-space selection problem."""

    key: str
    atoms: tuple[tuple[str, float, float, float], ...]
    basis: str
    charge: int
    spin: int
    point_group: str
    target_labels: tuple[str, ...]
    target_electrons: int
    target_orbitals: int
    avas_threshold: float = 0.2
    observation_weight: float = 0.85
    cardinality_penalty: float = 10.0
    scf_conv_tol: float = 1.0e-10

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CaseSpec":
        molecule = _mapping(payload, "molecule")
        active_space = _mapping(payload, "active_space")
        selection = _mapping(payload, "selection")
        atoms = tuple(
            (
                str(atom[0]),
                float(atom[1]),
                float(atom[2]),
                float(atom[3]),
            )
            for atom in molecule["atoms"]
        )
        return cls(
            key=str(payload["key"]),
            atoms=atoms,
            basis=str(molecule["basis"]),
            charge=int(molecule.get("charge", 0)),
            spin=int(molecule.get("spin", 0)),
            point_group=str(molecule.get("point_group", "c1")),
            target_labels=tuple(str(label) for label in active_space["target_labels"]),
            target_electrons=int(active_space["electrons"]),
            target_orbitals=int(active_space["orbitals"]),
            avas_threshold=float(selection.get("avas_threshold", 0.2)),
            observation_weight=float(selection.get("observation_weight", 0.85)),
            cardinality_penalty=float(selection.get("cardinality_penalty", 10.0)),
            scf_conv_tol=float(selection.get("scf_conv_tol", 1.0e-10)),
        )

    @property
    def atom_string(self) -> str:
        return "; ".join(
            f"{symbol} {x:.12g} {y:.12g} {z:.12g}"
            for symbol, x, y, z in self.atoms
        )


@dataclass
class RunArtifacts:
    """Serializable result plus the arrays needed for cross-run comparisons."""

    result: dict[str, Any]
    mf: Any
    mo_coeff: np.ndarray
    active_indices: tuple[int, ...]


def solve_fixed_budget_qubo(
    scores: Sequence[float],
    target_orbitals: int,
    *,
    cardinality_penalty: float = 10.0,
) -> dict[str, Any]:
    """Solve the unary fixed-budget QUBO and return one complete set action.

    Under the hard constraint ``sum(x) == k``, minimizing

        -sum(score_i * x_i) + lambda * (sum(x) - k)^2

    is exactly equivalent to taking the k largest scores.  This is an exact
    group solution, not a sequence of one-orbital swaps.
    """

    values = np.asarray(scores, dtype=float).reshape(-1)
    k = int(target_orbitals)
    if k < 0 or k > values.size:
        raise ValueError(f"Cannot select {k} orbitals from {values.size} scores")
    ranked = sorted(range(values.size), key=lambda index: (-values[index], index))
    selected = tuple(sorted(int(index) for index in ranked[:k]))
    cardinality_error = len(selected) - k
    energy = (
        -float(np.sum(values[list(selected)]))
        + float(cardinality_penalty) * cardinality_error**2
    )
    return {
        "solver": "exact_fixed_budget_top_k",
        "objective": "-sum(p_i*x_i) + lambda*(sum(x_i)-k)^2",
        "target_cardinality": k,
        "cardinality_penalty": float(cardinality_penalty),
        "selected_indices": selected,
        "qubo_energy": energy,
    }


def run_case(
    payload: Mapping[str, Any],
    *,
    use_symmetry: bool = False,
) -> RunArtifacts:
    """Run one cold-start active-space control."""

    spec = CaseSpec.from_mapping(payload)
    mf = _run_scf(spec, use_symmetry=use_symmetry)
    observation = _avas_observation(mf, spec)

    nmo = int(observation["scores"].size)
    prior_value = float(spec.target_orbitals) / max(nmo, 1)
    prior = np.full(nmo, prior_value, dtype=float)
    posterior = (
        (1.0 - spec.observation_weight) * prior
        + spec.observation_weight * observation["scores"]
    )
    posterior = np.clip(posterior, 0.0, 1.0)

    action = solve_fixed_budget_qubo(
        posterior,
        spec.target_orbitals,
        cardinality_penalty=spec.cardinality_penalty,
    )
    selected = tuple(int(index) for index in action["selected_indices"])
    occupations = _occupation_numbers(mf, observation["mo_coeff"])
    energies = _orbital_energies(mf, observation["mo_coeff"])
    target_projection, target_ao_indices = _lowdin_projection_by_labels(
        mf,
        observation["mo_coeff"],
        spec.target_labels,
    )
    selected_occupations = occupations[list(selected)]
    selected_electrons = float(np.sum(selected_occupations))
    occupation_bins = _occupation_bins(selected_occupations)
    active_ncore = max((int(mf.mol.nelectron) - spec.target_electrons) // 2, 0)
    expected_active_block = tuple(
        range(active_ncore, active_ncore + spec.target_orbitals)
    )
    avas_indices = tuple(int(index) for index in observation["active_indices"])

    checks: dict[str, bool] = {
        "scf_converged": bool(mf.converged),
        "qubo_budget_exact": len(selected) == spec.target_orbitals,
        "avas_orbital_count_matches_target": int(observation["ncas"])
        == spec.target_orbitals,
        "avas_electron_count_matches_target": _electron_total(observation["nelecas"])
        == spec.target_electrons,
        "selected_electron_count_matches_target": abs(
            selected_electrons - spec.target_electrons
        )
        <= 1.0e-6,
        "complete_avas_target_recovered": set(selected) == set(avas_indices),
        "selected_block_is_post_core_active_block": selected == expected_active_block,
        "all_selected_avas_scores_pass_threshold": all(
            float(observation["scores"][index]) + 1.0e-12 >= spec.avas_threshold
            for index in selected
        ),
    }
    expected_bins = _mapping(payload, "claim_expectation").get("occupation_bins")
    if isinstance(expected_bins, Mapping):
        checks["occupation_pattern_matches_claim"] = all(
            occupation_bins.get(str(name), 0) == int(count)
            for name, count in expected_bins.items()
        )

    table = []
    avas_set = set(avas_indices)
    selected_set = set(selected)
    for index in range(nmo):
        table.append(
            {
                "mo": index,
                "selected": index in selected_set,
                "avas_candidate": index in avas_set,
                "avas_score": float(observation["scores"][index]),
                "posterior_belief": float(posterior[index]),
                "target_ao_projection": float(target_projection[index]),
                "occupation": float(occupations[index]),
                "fock_diagonal_hartree": float(energies[index]),
            }
        )

    result = {
        "schema": "minimal_valence_control_result_v1",
        "key": spec.key,
        "design_role": str(payload.get("design_role", "")),
        "purpose": str(payload.get("purpose", "")),
        "claim_tier": str(payload.get("claim_tier", "")),
        "route": {
            "symmetry_enabled": bool(use_symmetry),
            "requested_point_group": spec.point_group if use_symmetry else "c1",
            "detected_point_group": str(getattr(mf.mol, "groupname", "C1")),
        },
        "molecule": {
            "atom": spec.atom_string,
            "basis": spec.basis,
            "charge": spec.charge,
            "spin": spec.spin,
            "electron_count": int(mf.mol.nelectron),
            "scf_method": "ROHF" if spec.spin else "RHF",
            "scf_energy_hartree": float(mf.e_tot),
            "scf_converged": bool(mf.converged),
        },
        "active_space_target": {
            "electrons": spec.target_electrons,
            "orbitals": spec.target_orbitals,
            "ao_labels": list(spec.target_labels),
        },
        "selection_pipeline": [
            "PySCF SCF",
            "AVAS target observation",
            "POMDP belief update",
            "one exact fixed-budget QUBO set action",
        ],
        "avas_observation": {
            "threshold": spec.avas_threshold,
            "reported_ncas": int(observation["ncas"]),
            "reported_nelecas": _json_electrons(observation["nelecas"]),
            "candidate_indices": list(avas_indices),
            "occ_weights": observation["occ_weights"].tolist(),
            "vir_weights": observation["vir_weights"].tolist(),
            "target_ao_indices": target_ao_indices,
            "mo_basis_kind": "avas_observation_mo",
            "mo_coeff_source": "pyscf.mcscf.avas.AVAS.kernel",
        },
        "pomdp": {
            "prior_probability": prior_value,
            "observation_weight": spec.observation_weight,
            "posterior_entropy": _binary_entropy(posterior),
        },
        "qubo_action": {
            **action,
            "selected_indices": list(selected),
            "proposal_kind": "complete_active_orbital_set",
        },
        "selected_active_space": {
            "indices": list(selected),
            "index_basis": "avas_observation_mo",
            "selected_electron_sum": selected_electrons,
            "occupation_bins": occupation_bins,
            "post_core_block": list(expected_active_block),
            "projection_score": float(
                np.sum(observation["scores"][list(selected)])
            ),
            "minimum_selected_target_ao_projection": float(
                np.min(target_projection[list(selected)])
            ),
        },
        "checks": checks,
        "passed": all(checks.values()),
        "orbital_table": table,
    }
    return RunArtifacts(
        result=result,
        mf=mf,
        mo_coeff=np.asarray(observation["mo_coeff"], dtype=float),
        active_indices=selected,
    )


def compare_subspaces(
    left: RunArtifacts,
    right: RunArtifacts,
    *,
    energy_tolerance: float,
    minimum_singular_value: float,
) -> dict[str, Any]:
    """Compare two selected spaces without relying on MO index equality."""

    if len(left.active_indices) != len(right.active_indices):
        raise ValueError("Subspaces have different dimensions")
    left_coeff = left.mo_coeff[:, list(left.active_indices)]
    right_coeff = right.mo_coeff[:, list(right.active_indices)]
    overlap = np.asarray(left.mf.get_ovlp(), dtype=float)
    cross_overlap = left_coeff.T @ overlap @ right_coeff
    singular_values = np.linalg.svd(cross_overlap, compute_uv=False)
    dimension = len(left.active_indices)
    projector_distance = float(
        np.sqrt(max(0.0, 2.0 * dimension - 2.0 * np.sum(singular_values**2)))
    )
    energy_difference = float(
        right.result["molecule"]["scf_energy_hartree"]
        - left.result["molecule"]["scf_energy_hartree"]
    )
    checks = {
        "same_target_electrons": left.result["active_space_target"]["electrons"]
        == right.result["active_space_target"]["electrons"],
        "same_target_orbitals": left.result["active_space_target"]["orbitals"]
        == right.result["active_space_target"]["orbitals"],
        "same_target_labels": left.result["active_space_target"]["ao_labels"]
        == right.result["active_space_target"]["ao_labels"],
        "both_routes_pass_case_checks": bool(left.result["passed"])
        and bool(right.result["passed"]),
        "scf_energies_match": abs(energy_difference) <= float(energy_tolerance),
        "selected_subspaces_match": bool(
            singular_values.size
            and float(np.min(singular_values)) >= float(minimum_singular_value)
        ),
    }
    return {
        "energy_difference_right_minus_left_hartree": energy_difference,
        "energy_tolerance_hartree": float(energy_tolerance),
        "principal_singular_values": singular_values.tolist(),
        "minimum_principal_singular_value": float(np.min(singular_values)),
        "minimum_singular_value_required": float(minimum_singular_value),
        "projector_frobenius_distance": projector_distance,
        "active_indices_equal": left.active_indices == right.active_indices,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _run_scf(spec: CaseSpec, *, use_symmetry: bool) -> Any:
    from pyscf import gto, scf

    molecule = gto.M(
        atom=spec.atom_string,
        basis=spec.basis,
        unit="Angstrom",
        charge=spec.charge,
        spin=spec.spin,
        symmetry=spec.point_group if use_symmetry else False,
        verbose=0,
    )
    mf = scf.ROHF(molecule) if spec.spin else scf.RHF(molecule)
    mf.conv_tol = spec.scf_conv_tol
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"SCF did not converge for {spec.key}")
    return mf


def _avas_observation(mf: Any, spec: CaseSpec) -> dict[str, Any]:
    from pyscf.mcscf import avas

    avas_obj = avas.AVAS(
        mf,
        list(spec.target_labels),
        threshold=spec.avas_threshold,
        minao="minao",
        with_iao=True,
        openshell_option=2,
        canonicalize=True,
        ncore=0,
        verbose=0,
    )
    ncas, nelecas, mo_coeff = avas_obj.kernel()
    mo_coeff = np.asarray(mo_coeff, dtype=float)
    occ_weights = np.asarray(avas_obj.occ_weights, dtype=float)
    vir_weights = np.asarray(avas_obj.vir_weights, dtype=float)
    scores = np.zeros(mo_coeff.shape[1], dtype=float)
    occ_active_count = int(np.count_nonzero(occ_weights >= spec.avas_threshold))
    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) != 0))
    inactive_occ_count = max(nocc - occ_active_count, 0)
    scores[: occ_weights.size] = occ_weights
    scores[occ_weights.size : occ_weights.size + vir_weights.size] = vir_weights
    active_indices = tuple(
        range(inactive_occ_count, inactive_occ_count + int(ncas))
    )
    return {
        "ncas": int(ncas),
        "nelecas": nelecas,
        "mo_coeff": mo_coeff,
        "scores": np.clip(scores, 0.0, 1.0),
        "active_indices": active_indices,
        "occ_weights": occ_weights,
        "vir_weights": vir_weights,
    }


def _lowdin_projection_by_labels(
    mf: Any,
    mo_coeff: np.ndarray,
    labels: Sequence[str],
) -> tuple[np.ndarray, list[int]]:
    ao_indices: set[int] = set()
    for label in labels:
        ao_indices.update(int(index) for index in mf.mol.search_ao_label(str(label)))
    selected_ao_indices = sorted(ao_indices)
    if not selected_ao_indices:
        raise ValueError(f"No AO labels matched {tuple(labels)!r}")
    overlap = np.asarray(mf.get_ovlp(), dtype=float)
    eigvals, eigvecs = np.linalg.eigh(overlap)
    sqrt_overlap = (
        eigvecs
        @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None)))
        @ eigvecs.T
    )
    orthogonal_ao_coeff = sqrt_overlap @ np.asarray(mo_coeff, dtype=float)
    projection = np.sum(
        orthogonal_ao_coeff[selected_ao_indices, :] ** 2,
        axis=0,
    )
    return np.clip(projection, 0.0, 1.0), selected_ao_indices


def _occupation_numbers(mf: Any, mo_coeff: np.ndarray) -> np.ndarray:
    overlap = np.asarray(mf.get_ovlp(), dtype=float)
    density = np.asarray(mf.make_rdm1(), dtype=float)
    if density.ndim == 3:
        density = np.sum(density, axis=0)
    coeff = np.asarray(mo_coeff, dtype=float)
    return np.diag(coeff.T @ overlap @ density @ overlap @ coeff).reshape(-1)


def _orbital_energies(mf: Any, mo_coeff: np.ndarray) -> np.ndarray:
    fock = np.asarray(mf.get_fock(), dtype=float)
    if fock.ndim == 3:
        fock = fock[0]
    coeff = np.asarray(mo_coeff, dtype=float)
    return np.diag(coeff.T @ fock @ coeff).reshape(-1)


def _occupation_bins(occupations: Sequence[float]) -> dict[str, int]:
    values = np.asarray(occupations, dtype=float)
    return {
        "doubly_occupied": int(np.count_nonzero(values > 1.5)),
        "singly_occupied": int(
            np.count_nonzero((values >= 0.5) & (values <= 1.5))
        ),
        "empty": int(np.count_nonzero(values < 0.5)),
    }


def _binary_entropy(probabilities: Sequence[float]) -> float:
    values = np.clip(np.asarray(probabilities, dtype=float), 1.0e-12, 1.0 - 1.0e-12)
    return float(
        -np.sum(values * np.log(values) + (1.0 - values) * np.log(1.0 - values))
    )


def _electron_total(value: Any) -> int:
    if isinstance(value, (tuple, list)):
        return sum(int(item) for item in value)
    return int(value)


def _json_electrons(value: Any) -> int | list[int]:
    if isinstance(value, (tuple, list)):
        return [int(item) for item in value]
    return int(value)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key!r} must be a JSON object")
    return value
