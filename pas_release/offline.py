"""Offline selection from fixed imported observations, without solver dispatch."""

import csv
import shutil
from pathlib import Path

import numpy as np

from pas import metrics, qubo
from pas.belief import Quadratic
from pas.evidence_belief import statistic_quadratic

from .common import interface_identity, require, verify_source, write_new
from .offline_input import load
from .resources import PACKAGE

EXAMPLE = PACKAGE / "offline_example"


def example(output):
    output = Path(output).resolve()
    require(not output.exists(), "Choose a new example directory: " + str(output))
    require(
        not output.is_relative_to(PACKAGE), "Examples must be copied outside the installed package"
    )
    shutil.copytree(EXAMPLE, output)
    return dict(
        mode="offline_example",
        directory=str(output),
        summary="Example created: "
        + str(output)
        + "\nRun: pas offline select "
        + str(output / "input.json")
        + " --output "
        + str(output / "result"),
    )


def objective(spec):
    """Reuse the core's terminal-scale input statistics and determinant QUBO."""
    context, cfg = spec.context, spec.config
    weights = np.asarray(context.states["weights"])
    result = Quadratic(0.0, np.zeros(context.nmo), np.zeros((context.nmo, context.nmo)))
    total = sum(cfg["weights"].values())
    for channel in cfg["channels"]:
        if channel == "composition":
            component = metrics.composition_quadratic(
                context.composition,
                context.active,
                context.composition_pair,
                "determinant_pair_v1",
                composition_purity_weight=cfg["composition_purity_weight"],
            )
            part = Quadratic(**component)
        else:
            if channel == "occupation":
                diagonal = np.diagonal(spec.densities, axis1=1, axis2=2)
                first, second = np.triu_indices(len(spec.measured), 1)
                means = dict(
                    u=weights @ (2 * diagonal - diagonal**2),
                    b=np.einsum("r,rij->ij", weights, abs(spec.densities) ** 2)[first, second],
                )
            else:
                first, second = np.triu_indices(len(spec.measured), 1)
                means = dict(
                    h=weights @ spec.entropy,
                    m=np.einsum("r,rij->ij", weights, spec.mutual)[first, second],
                )
            linear, pair = statistic_quadratic(
                context,
                spec.measured,
                means,
                dict(entropy=1.0, mutual_information=1.0),
                channel,
                cfg,
            )
            part = Quadratic(0.0, linear, pair)
        result = result + part.scaled(cfg["weights"][channel] / total)
    return result


def score(spec, active):
    """Report observed subspace descriptors, never a new target-CAS energy."""
    context, cfg = spec.context, spec.config
    positions = {orbital: i for i, orbital in enumerate(spec.measured)}
    selected = [positions[i] for i in active]
    densities = spec.densities[:, selected][:, :, selected] if spec.densities is not None else None
    entropy = spec.entropy[:, selected] if spec.entropy is not None else None
    mutual = spec.mutual[:, selected][:, :, selected] if spec.mutual is not None else None
    values = metrics.evaluate(
        cfg["channels"],
        cfg["weights"],
        densities=densities,
        state_weights=context.states["weights"],
        composition=context.composition,
        active=active,
        entropies=entropy,
        mutual_information=mutual,
        composition_pair=context.composition_pair,
        composition_model="determinant_pair_v1",
        composition_purity_weight=cfg["composition_purity_weight"],
    )
    boundary = 0.0
    if mutual is not None:
        outside = [i for i in range(len(spec.measured)) if i not in selected]
        _, denominator = metrics.qit_normalizers(context.k)
        per_state = spec.mutual[:, selected][:, :, outside].sum(axis=(1, 2))
        boundary = float(np.asarray(context.states["weights"]) @ per_state / denominator)
    proposal = (
        values["quality"] - cfg["weights"].get("qit", 0.0) / sum(cfg["weights"].values()) * boundary
    )
    return dict(
        active_indices=[i + 1 for i in active],
        components=values,
        qit_boundary_penalty=boundary,
        selection_score=proposal,
    )


def check(input_file):
    source_id = verify_source()
    spec = load(input_file)
    count = qubo.candidate_count(spec.context)
    return dict(
        mode="offline_check",
        passed=True,
        scientific_source_id=source_id,
        name=spec.raw["name"],
        input_files=spec.files,
        evidence=spec.config["channels"],
        measured_orbitals=[i + 1 for i in spec.measured],
        legal_candidates=count,
        audit=spec.audit,
        effective_settings=spec.config,
        summary=f"Input valid: {spec.raw['name']}\nMeasured orbitals: {len(spec.measured)}; legal candidates: {count}\nEvidence: {', '.join(spec.config['channels'])}\nNo electronic-structure solver is called.",
    )


def select(input_file, output):
    source_id = verify_source()
    output = Path(output).resolve()
    require(not output.exists(), "Choose a new result directory: " + str(output))
    require(not output.is_relative_to(PACKAGE), "Results must be outside the installed package")
    spec = load(input_file)
    quadratic = objective(spec)
    winner, solver = qubo.solve(
        spec.context, quadratic, spec.config, spec.config["seed"], collect=spec.config["candidates"]
    )
    require(winner is not None, "No feasible candidate satisfies the supplied constraints")
    # Retain the user's initial choice even if stochastic search did not visit it.
    candidates = {tuple(a) for a in solver["visited_candidates"]} | {
        tuple(sorted(spec.context.active)),
        tuple(winner),
    }
    ranked = sorted(
        (score(spec, a) for a in candidates),
        key=lambda r: (-r["selection_score"], r["active_indices"]),
    )
    score_errors = []
    # The core accepts density eigenvalues within 1e-7 of [0, 2] and clips
    # terminal descriptors to [0, 1]; a raw quadratic can differ at that edge.
    score_tolerance = 1e-6
    for row in ranked:
        active = [i - 1 for i in row["active_indices"]]
        score_errors.append(abs(quadratic(active) - row["selection_score"]))
        require(
            score_errors[-1] <= score_tolerance,
            "QUBO objective and independently evaluated evidence score differ",
        )
    initial = score(spec, spec.context.active)
    best = ranked[0]
    spec.audit["score_evaluation_max_error"] = max(score_errors)
    spec.audit["score_evaluation_tolerance"] = score_tolerance
    solver = {
        **solver,
        "visited_candidates": [[i + 1 for i in row] for row in solver["visited_candidates"]],
    }
    result = dict(
        schema="pas.offline.result.v1",
        mode="offline_select",
        name=spec.raw["name"],
        source=spec.raw["source"],
        scientific_source_id=source_id,
        interface_id=interface_identity(),
        score_model="fixed_orbital_terminal_descriptors_with_qit_boundary_v1",
        evidence=spec.config["channels"],
        effective_settings=spec.config,
        initial=initial,
        recommended=best,
        candidates=ranked,
        score_change=best["selection_score"] - initial["selection_score"],
        solver=solver,
        input_files=spec.files,
        audit=spec.audit,
        new_quantum_chemistry_calculations=0,
        optimized_candidate_energy_hartree=None,
        result_directory=str(output),
    )
    result["summary"] = (
        f"PAS offline selection: {spec.raw['name']}\n"
        f"Evidence: {', '.join(spec.config['channels'])}\n"
        f"Active space: {spec.context.electrons} electrons / {spec.context.k} orbitals\n"
        f"Initial orbitals (1-based): {initial['active_indices']}\n"
        f"Recommended orbitals (1-based): {best['active_indices']}\n"
        f"Initial selection score: {initial['selection_score']:.12g}\n"
        f"Recommended selection score: {best['selection_score']:.12g}\n"
        f"Components: {best['components']}\n"
        f"Global optimum proven: {solver['global_optimum_proven']}\n"
        "Scores describe the imported fixed-orbital observations. No new CAS energy or orbital optimization was computed.\n"
    )
    spec.unchanged()
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "result.json", result)
    write_new(output / "input.json", spec.raw)
    (output / "summary.txt").write_text(result["summary"], encoding="utf-8")
    columns = [
        "rank",
        "active_indices",
        *spec.config["channels"],
        "quality",
        "qit_boundary_penalty",
        "selection_score",
    ]
    with (output / "candidates.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for rank, row in enumerate(ranked, 1):
            writer.writerow(
                dict(
                    rank=rank,
                    active_indices=" ".join(map(str, row["active_indices"])),
                    **row["components"],
                    qit_boundary_penalty=row["qit_boundary_penalty"],
                    selection_score=row["selection_score"],
                )
            )
    np.savez_compressed(
        output / "objective.npz",
        constant=quadratic.constant,
        linear=quadratic.linear,
        pair=quadratic.pair,
    )
    return result
