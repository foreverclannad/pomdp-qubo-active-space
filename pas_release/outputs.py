"""Stable user-facing result files derived from immutable engine records."""

import csv
import io
import os
from pathlib import Path

from pas.common import write_json
from pas.controller import source_identity

from .common import SOURCE, read_json, require, sha

HISTORY_FIELDS = (
    "case",
    "event",
    "kind",
    "valid",
    "accepted",
    "input_active_indices",
    "quality",
    "energy_hartree",
    "cpu_seconds",
    "wall_seconds",
    "context_id",
    "composition_alpha",
    "historical_quality",
)


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _root_energies(receipt):
    artifact = receipt.get("artifacts", {}).get("mps_evidence")
    if artifact:
        require(sha(artifact["path"]) == artifact["sha256"], "Terminal MPS evidence changed")
        return read_json(artifact["path"]).get("root_energies")
    context = receipt.get("output_context", {})
    # DMRG orbital containers can carry earlier energies; never substitute them.
    if context.get("backend") == "cas":
        path = context.get("payload", {}).get("basis_file")
        if path:
            require(sha(path) == context["basis_id"], "Final orbital basis changed")
            import h5py

            with h5py.File(path, "r") as handle:
                return handle["ROOT_ENERGIES"][()].reshape(-1).astype(float).tolist()
    return None


def state_result(state, spec, interface_id, engine_directory):
    context, incumbent = state["context"], state["incumbent"]
    metrics = incumbent.get("metrics", {})
    result = dict(
        schema="pas.result.v1",
        mode="calculation",
        name=spec["name"],
        profile=spec["profile"],
        status=state["status"],
        termination=state.get("termination"),
        scientific_baseline=SOURCE,
        scientific_source_id=source_identity(),
        interface_id=interface_id,
        config_id=spec["config_id"],
        orbital_index_base=1,
        units=dict(energy="hartree", cpu_time="second", wall_time="second", scores="dimensionless"),
        active_space=dict(
            electrons=context["electrons"],
            orbitals=len(context["active"]),
            indices=[i + 1 for i in context["active"]],
        ),
        states=context["states"],
        metrics=metrics,
        root_energies_hartree=_root_energies(incumbent),
        convergence=dict(
            valid=incumbent.get("valid", False),
            converged=incumbent.get("converged", False),
            state_valid=incumbent.get("state_valid", False),
        ),
        actions=len(state["events"]),
        accepted_actions=sum(e["accepted"] for e in state["events"]),
        remaining_budget=dict(
            diagnostic=state["remaining_diagnostics"], optimization=state["remaining_optimizations"]
        ),
        search_cost=dict(
            cpu_seconds=state["search_cpu_seconds"], wall_seconds=state["search_wall_seconds"]
        ),
        pending=state["pending"] is not None,
        engine_records="engine/",
        historical_profile_unchanged=spec["historical_profile_unchanged"],
        paper_profile_unchanged=spec["paper_profile_unchanged"],
        composition_purity_weight=spec["config"].get("composition_purity_weight", 0.5),
    )
    rows = []
    for event in state["events"]:
        path = Path(engine_directory) / event["receipt"]
        require(sha(path) == event["sha256"], "Event receipt changed: " + str(path))
        receipt = read_json(path)
        rows.append(
            _history_row(spec["profile"], event, receipt, result["composition_purity_weight"])
        )
    return result, rows


def _history_row(case, event, receipt, alpha):
    values = receipt.get("metrics", {})
    return dict(
        case=case,
        event=event["number"],
        kind=receipt["kind"],
        valid=receipt["valid"],
        accepted=event["accepted"],
        input_active_indices=" ".join(str(i + 1) for i in receipt["active"]),
        quality=values.get("quality"),
        energy_hartree=values.get("energy"),
        cpu_seconds=receipt.get("cpu_seconds"),
        wall_seconds=receipt.get("wall_seconds"),
        context_id=receipt["context_id"],
        composition_alpha=alpha,
    )


def summary_text(result):
    lines = [
        "PAS result",
        "Name: " + result.get("name", "paper acceptance"),
        "Status: " + result["status"],
    ]
    if result.get("mode") == "calculation":
        space = result["active_space"]
        lines += [
            f"Active space: {space['electrons']} electrons in {space['orbitals']} orbitals",
            "Orbitals (1-based): " + " ".join(map(str, space["indices"])),
            f"Actions: {result['actions']} ({result['accepted_actions']} accepted)",
        ]
        for key, value in result["metrics"].items():
            if value is not None:
                lines.append(f"{key}{' / hartree' if key == 'energy' else ''}: {value:.12g}")
        lines.append("Incumbent converged: " + str(result["convergence"]["converged"]))
        lines.append(f"Composition alpha: {result['composition_purity_weight']:g}")
    elif result.get("mode") == "replay":
        lines += [
            "Frozen evidence and small-molecule acceptance passed.",
            "No new dio/TiF3 OpenMolcas or Block2 calculation.",
        ]
        for name, case in result.get("cases", {}).items():
            metrics = case["metrics"]
            lines.append(f"{name}: energy = {metrics['energy']:.12f} hartree")
            if "quality" in metrics:
                lines.append(f"  Quality = {metrics['quality']:.10f}")
            if "composition_purity_weight" in case:
                lines.append(f"  Composition alpha = {case['composition_purity_weight']:g}")
                if (
                    case["historical_composition_purity_weight"]
                    != case["composition_purity_weight"]
                ):
                    lines.append(
                        f"  Historical quality (alpha={case['historical_composition_purity_weight']:g}) = {case['historical_metrics']['quality']:.10f}"
                    )
            if "projection_sum" in metrics:
                lines.append(f"  Projection sum = {metrics['projection_sum']:.4f}")
            lines.append("  Orbitals (1-based): " + " ".join(map(str, case["active_indices"])))
    if result.get("error"):
        lines.append("Error: " + result["error"])
    return "\n".join(lines) + "\n"


def write_results(directory, result, history=()):
    directory = Path(directory)
    write_json(directory / "result.json", result)
    atomic_text(directory / "summary.txt", summary_text(result))
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=HISTORY_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(history)
    atomic_text(directory / "history.csv", buffer.getvalue())


def replay_results(evidence_directory, report_directory):
    from .common import Evidence
    from .resources import CASE_NAMES, DATA

    evidence = Evidence(evidence_directory)
    profiles = read_json(DATA / "cases.json")["cases"]
    scoring = read_json(Path(report_directory) / "scoring_changes.json")["cases"]
    physical = {}
    for group in ("dio", "native"):
        physical.update(read_json(Path(report_directory) / f"physical_{group}.json")["runs"])
    cases, history = {}, []
    for public_name, key in CASE_NAMES.items():
        record = physical[key]
        scored = scoring[public_name]
        cases[public_name] = dict(
            metrics=scored["final"]["paper_metrics"],
            historical_metrics=scored["final"]["historical_metrics"],
            scoring_changes=scored["final"]["changes"],
            root_energies_hartree=record["final"]["root_energies_hartree"],
            active_indices=[i + 1 for i in record["final"]["output_active"]],
            orbital_index_base=1,
            composition_purity_weight=scored["paper_alpha"],
            historical_composition_purity_weight=scored["historical_alpha"],
            scoring_source="recomputed_from_frozen_physical_evidence",
            reference_checks=record["reference_checks"],
        )
        directory = Path(profiles[key]["original_run"])
        state = evidence.read(directory / "state.json")
        scored_events = {row["event"]: row for row in scored["events"]}
        for event in state["events"]:
            receipt = evidence.read(directory / event["receipt"], event["sha256"])
            values = receipt.get("metrics", {})
            paper_values = scored_events.get(event["number"], {}).get("paper_metrics", values)
            row = _history_row(public_name, event, receipt, scored["paper_alpha"])
            row.update(
                quality=paper_values.get("quality"), historical_quality=values.get("quality")
            )
            history.append(row)
    evidence.unchanged()
    return cases, history


def small_molecule_results(report):
    cases = {}
    for name, row in report["cases"].items():
        result = row["result"]
        selected = result["selected_active_space"]
        cases[name] = dict(
            metrics=dict(
                energy=result["molecule"]["scf_energy_hartree"],
                projection_sum=selected["projection_score"],
            ),
            active_indices=[i + 1 for i in selected["indices"]],
            orbital_index_base=1,
        )
    return cases
