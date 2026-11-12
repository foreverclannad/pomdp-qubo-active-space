"""Reconstruct physical target metrics from their original bound arrays."""

import h5py
import numpy as np

from pas import backend, metrics
from pas.common import identity
from pas.composition import determinant_pair_matrix
from pas.contracts import Context
from pas.input_evidence import diagnostic_arrays
from pas.orbital_io import (
    ao_subspace_projection_scores,
    read_openmolcas_orbital_hdf5,
    select_ao_indices,
)

from ..common import read_json, require


def audit_target(evidence, receipt, cfg):
    require(
        receipt["evidence_id"]
        == identity({k: v for k, v in receipt.items() if k != "evidence_id"}),
        "Target receipt content identity changed",
    )
    require(
        all((receipt[k] for k in ("valid", "converged", "state_valid"))),
        "Target receipt must be valid, converged and state-valid",
    )
    ctx = Context.from_dict(receipt["output_context"])
    for v in receipt["artifacts"].values():
        if isinstance(v, dict) and {"path", "sha256"} <= v.keys():
            evidence.artifact(v)
    mo_path = evidence.bind(ctx.payload["basis_file"], ctx.basis_id)
    evidence.bind(ctx.payload["checkpoint"], ctx.payload["checkpoint_hash"])
    mo = read_openmolcas_orbital_hdf5(mo_path)
    rules = ctx.payload["case"]["target"]["rules"]
    projection = ao_subspace_projection_scores(
        mo.mo_coefficients, mo.ao_overlap, select_ao_indices(mo, rules)
    )
    pair, _ = determinant_pair_matrix(mo, rules)
    ao_errors = dict(
        projection=float(np.max(abs(projection - np.asarray(ctx.composition)))),
        pair=float(np.max(abs(pair - np.asarray(ctx.composition_pair)))),
    )
    require(max(ao_errors.values()) < 1e-12, ao_errors)
    density = np.load(evidence.artifact(receipt["artifacts"]["densities"]), allow_pickle=False)
    entropy = mutual = None
    if ctx.backend == "cas":
        raw_density, energies = backend.read_densities(mo_path, ctx.states, ctx.electrons)
        endpoint = "ROOT_ENERGIES in signed CASSCF HDF5"
    else:
        raw = read_json(evidence.artifact(receipt["artifacts"]["mps_evidence"]))
        require(
            raw["status"] == "available",
            "Terminal MPS evidence is unavailable",
        )
        require(
            raw["entropy_unit"] == "natural_log",
            "Terminal MPS entropy must use natural logarithms",
        )
        require(
            raw["mutual_information_convention"] == "0.5*(s_i+s_j-s_ij)",
            "Terminal MPS mutual information must use 0.5*(s_i+s_j-s_ij)",
        )
        raw_density, entropy, mutual = diagnostic_arrays(raw, ctx)
        energies = np.asarray(raw["root_energies"], float)
        endpoint = "root_energies in signed terminal MPS evidence"
    density_error = float(np.max(abs(raw_density - density)))
    require(density_error < 1e-12, density_error)
    kw = dict(
        densities=density,
        state_weights=ctx.states["weights"],
        electrons=ctx.electrons,
        composition=ctx.composition,
        active=ctx.active,
        entropies=entropy,
        mutual_information=mutual,
        composition_pair=ctx.composition_pair,
        composition_channels=ctx.composition_channels,
        composition_model=cfg["composition_model"],
        composition_preferences=ctx.composition_preferences,
        composition_purity_weight=cfg.get("composition_purity_weight", 0.5),
    )
    values = metrics.evaluate(cfg["channels"], cfg["weights"], **kw)
    values["energy"] = float(energies @ ctx.states["weights"])
    errors = {k: abs(v - receipt["metrics"][k]) for k, v in values.items()}
    require(max(errors.values()) < 1e-12, errors)
    with h5py.File(mo_path, "r") as h:
        container = np.asarray(h["ROOT_ENERGIES"][()], float).reshape(-1)
    row = dict(
        evidence_id=receipt["evidence_id"],
        metrics=values,
        metric_errors=errors,
        density_max_error=density_error,
        ao_reconstruction_errors=ao_errors,
        root_energies_hartree=energies.tolist(),
        energy_endpoint=endpoint,
        orbital_container_energy_max_difference=float(np.max(abs(energies - container))),
        output_active=list(ctx.active),
        basis_sha256=ctx.basis_id,
    )
    if len(energies) == 2:
        row["gap_eV"] = float((energies[1] - energies[0]) * 27.211386245988)
    if len(energies) == 5:
        row["gaps_cm_inverse"] = ((energies - energies[0]) * 219474.6313705).tolist()
        row["pair_means_cm_inverse"] = [
            float(np.mean(row["gaps_cm_inverse"][1:3])),
            float(np.mean(row["gaps_cm_inverse"][3:5])),
        ]
    if ctx.backend == "dmrg":
        kw["composition_purity_weight"] = 0.45
        row["rescored_alpha045"] = metrics.evaluate(cfg["channels"], cfg["weights"], **kw)
    return row
