"""paper scoring views and explicit differences from immutable historical receipts."""

import csv
from pathlib import Path

from pas.common import identity

from ..common import require
from ..profiles import PAPER_INPUTS, paper_configuration
from ..resources import CASE_NAMES


def scoring_view(target, historical_alpha, paper_alpha):
    historical = dict(target["metrics"])
    current = dict(historical)
    if historical_alpha != paper_alpha:
        require(paper_alpha == 0.45, "Only the validated paper alpha=0.45 rescore is available")
        current.update(target["rescored_alpha045"])
    for metric in ("occupation", "qit", "energy"):
        require(current.get(metric) == historical.get(metric), "Alpha changed " + metric)
    return dict(
        historical_metrics=historical,
        paper_metrics=current,
        changes={key: current[key] - historical[key] for key in ("composition", "quality")},
    )


def run(evidence, cases, physical_runs, alpha_replay):
    """Link public profiles/materials to the independently validated planner replay."""
    points = {point["alpha"]: point["runs"] for point in alpha_replay["points"]}
    result = dict(schema="pas.paper-scoring.v1", cases={}, physical_jobs_launched=0)
    legacy_names = {"NX1_DMRG": "NX1", "CI_OO_DMRG": "CI_OO_N00"}
    for public_name, key in CASE_NAMES.items():
        case, physical = cases[key], physical_runs[key]
        config = paper_configuration(case)
        paper_alpha = config["composition_purity_weight"]
        historical_alpha = physical["config"].get("composition_purity_weight", 0.5)
        manifest = evidence.read(Path(case["original_run"]) / "manifest.json")
        companions = {}
        if manifest.get("references"):
            companions["references_file"] = manifest["references"]
        if manifest.get("bootstrap_input_evidence"):
            saved = evidence.read(manifest["bootstrap_input_evidence"]["path"])
            companions["bootstrap_evidence_file"] = saved["source"]
        require(
            companions == PAPER_INPUTS[key],
            "Paper template omits or changes historical companion inputs: " + public_name,
        )
        for binding in companions.values():
            if binding["path"] in evidence.index:
                evidence.artifact(binding)
        if key in legacy_names:
            replay = points[paper_alpha][legacy_names[key]]
            require(
                replay["effective_config_id"] == identity(config), "Public/replay config differs"
            )
            require(replay["all_original_dispatches_and_acceptances_unchanged"], public_name)
        else:
            require(config == physical["config"], "Unexpected non-dio configuration change")
        rows = [
            dict(
                event=row["event"],
                kind=row["kind"],
                accepted=row.get("accepted"),
                **scoring_view(row, historical_alpha, paper_alpha),
            )
            for row in physical["rows"]
            if "metrics" in row
        ]
        result["cases"][public_name] = dict(
            historical_alpha=historical_alpha,
            paper_alpha=paper_alpha,
            paper_config_id=identity(config),
            companion_inputs=companions,
            companion_files_in_evidence_bundle={
                field: binding["path"] in evidence.index for field, binding in companions.items()
            },
            events=rows,
            final=scoring_view(physical["final"], historical_alpha, paper_alpha),
            physical_endpoint_unchanged=True,
            reference_checks=physical["reference_checks"],
        )
    evidence.unchanged()
    result["passed"] = True
    return result


def write_csv(path, report):
    fields = (
        "case",
        "event",
        "kind",
        "accepted",
        "historical_alpha",
        "paper_alpha",
        "composition_historical",
        "composition_paper",
        "composition_change",
        "quality_historical",
        "quality_paper",
        "quality_change",
        "energy_hartree",
    )
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for name, case in report["cases"].items():
            for event in case["events"]:
                row = dict(case=name, **{key: event[key] for key in ("event", "kind", "accepted")})
                row.update({key: case[key] for key in ("historical_alpha", "paper_alpha")})
                for metric in ("composition", "quality"):
                    row[metric + "_historical"] = event["historical_metrics"][metric]
                    row[metric + "_paper"] = event["paper_metrics"][metric]
                    row[metric + "_change"] = event["changes"][metric]
                row["energy_hartree"] = event["paper_metrics"]["energy"]
                writer.writerow(row)
