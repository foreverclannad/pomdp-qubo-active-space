"""Run the complete paper paper acceptance suite from immutable archived evidence."""

import platform
import subprocess
import sys
import time
from pathlib import Path

from pas import controller

from .. import RELEASE, verify
from ..common import SOURCE, Evidence, compare_values, read_json, require, write_new
from ..resources import DATA, PACKAGE

CORE_ROOT = Path(controller.__file__).resolve().parent.parent


def check_ci_cc(actual, expected):
    diagnostic_angles = {
        "principal_angles_radians",
        "principal_angles_degrees",
        "maximum_principal_angle_degrees",
    }

    def stable(record):
        return {
            **record,
            "comparison": {
                k: v for k, v in record["comparison"].items() if k not in diagnostic_angles
            },
        }

    return compare_values(stable(actual), stable(expected), location="CI_CC")


def check_results(physical_runs, planning_runs, controls):
    expected = read_json(DATA / "expected_results.json")["results"]
    cases = read_json(DATA / "cases.json")["cases"]
    errors = {}
    for name, run in physical_runs.items():
        target = expected[name]
        actual = {k: run["final"][k] for k in target if k != "references"}
        actual["references"] = {
            k: {f: run["reference_checks"][k][f] for f in value}
            for k, value in target["references"].items()
        }
        errors[name] = compare_values(actual, target, location=name)
        accepted = [row["accepted"] for row in run["rows"] if row["kind"] == "optimize"]
        require(accepted == cases[name]["optimization_acceptance"], name)
        require(len(run["rows"]) - 1 == cases[name]["events"], name)
    for name, run in planning_runs.items():
        require(
            len(run["rounds"]) == cases[name]["optimization_choices"],
            "Number of replayed optimization choices differs from the frozen case",
        )
        require(
            run["events_replayed"] == cases[name]["events"],
            "Number of replayed events differs from the frozen case",
        )
    for name in ("controls", "MAE_steps_2_to_4", "MAE_reduction_percent"):
        errors[name] = compare_values(controls[name], expected[name], location=name)
    errors["CI_CC"] = check_ci_cc(controls["CI_CC"], expected["CI_CC"])
    choices = {k: v["active"] for k, v in controls["fourth_decision"]["choices"].items()}
    require(
        choices == expected["control_choices"],
        "A/B/C orbital choices differ from the paper controls",
    )
    return errors


def tool(output, name, arguments):
    with (
        (output / (name + ".stdout.log")).open("x") as stdout,
        (output / (name + ".stderr.log")).open("x") as stderr,
    ):
        subprocess.run(
            [sys.executable, *map(str, arguments)], stdout=stdout, stderr=stderr, check=True
        )
    return read_json(output / (name + ".json"))


def run(evidence_directory, output):
    started = time.monotonic()
    evidence = Evidence(evidence_directory)
    output = Path(output).resolve()
    if output.is_relative_to(evidence.root):
        raise ValueError("Outputs must be outside the immutable evidence bundle")
    if output.is_relative_to(PACKAGE):
        raise ValueError("Outputs must not be placed inside release inputs")
    output.mkdir(parents=True, exist_ok=False)
    print("Verifying the frozen scientific source and evidence manifest", flush=True)
    verification = verify.run(evidence.root)
    write_new(output / "integrity_before.json", verification)
    from . import controls, physical, planning, scoring, small_molecules

    physical_runs = {}
    for group in ("dio", "native"):
        report = physical.run(evidence.root, group)
        physical_runs.update(report["runs"])
        write_new(output / ("physical_" + group + ".json"), report)
    print("Replaying fixed-pool A/B/C and CI-CC controls", flush=True)
    control_report = controls.run(evidence.root)
    write_new(output / "controls.json", control_report)
    cases = read_json(DATA / "cases.json")["cases"]
    planning_runs = {}
    for name in ("NX1_CAS", "CI_OO_CAS", "TiF3"):
        print("Replaying " + name, flush=True)
        planning_runs[name] = planning.replay(evidence.root, cases[name]["original_run"])
        write_new(output / ("planning_" + name + ".json"), planning_runs[name])
    print("Replaying legacy dio and the alpha=0.45 paper profile", flush=True)
    terminal = tool(
        output,
        "dio_terminal",
        [
            "-m",
            "pas_release.reproduction.legacy_dio",
            "--corrected-repo",
            CORE_ROOT,
            "--archive-root",
            evidence.tree,
            "--expected-source-id",
            verification["scientific_source_id"],
            "--output",
            output / "dio_terminal.json",
        ],
    )
    require(
        terminal["all_original_dispatches_reproduced"],
        "Legacy dio replay changed a recorded dispatch",
    )
    require(
        terminal["all_five_optimization_choices_reproduced"],
        "Legacy dio replay did not reproduce all five optimization choices",
    )
    require(
        terminal["all_three_original_13_of_13_results_reusable"],
        "Legacy dio replay changed a 13-of-13 reference result",
    )
    alpha = tool(
        output,
        "dio_alpha045",
        [
            "-m",
            "pas_release.reproduction.legacy_alpha",
            "--repo",
            CORE_ROOT,
            "--archive",
            evidence.tree,
            "--alphas",
            ".5",
            ".45",
            "--cases",
            "NX1",
            "CI_OO_N00",
            "--output",
            output / "dio_alpha045.json",
        ],
    )
    require(
        [p["alpha"] for p in alpha["points"]] == [0.5, 0.45],
        "Alpha comparison must cover the historical 0.5 and paper 0.45 settings",
    )
    require(
        all(
            (
                r["all_original_dispatches_and_acceptances_unchanged"]
                for point in alpha["points"]
                for r in point["runs"].values()
            )
        ),
        "Alpha=0.45 changed a recorded dispatch or acceptance decision",
    )
    errors = check_results(physical_runs, planning_runs, control_report)
    scored = scoring.run(evidence, cases, physical_runs, alpha)
    write_new(output / "scoring_changes.json", scored)
    scoring.write_csv(output / "scoring_changes.csv", scored)
    print("Recomputing five small-molecule controls and H2O symmetry", flush=True)
    small = small_molecules.run()
    write_new(output / "small_molecules.json", small)
    require(small["passed"], "Small-molecule acceptance failed")
    after = verify.run(evidence.root)
    require(
        after == verification, "Scientific source or evidence integrity changed during acceptance"
    )
    write_new(output / "integrity_after.json", after)
    import h5py
    import numpy
    import scipy

    report = dict(
        schema="pas_paper_acceptance_v1",
        passed=True,
        source_id=verification["scientific_source_id"],
        numerical_baseline_source_id=SOURCE,
        release=RELEASE,
        expected_result_max_errors=errors,
        main_search_histories=5,
        main_search_events=31,
        main_optimization_choices=19,
        main_target_endpoints_including_seeds=24,
        ABC_controls_and_CI_CC_passed=True,
        legacy_harmonic_control_passed=True,
        alpha045_dispatch_and_acceptance_unchanged=True,
        paper_profiles_and_companion_inputs_verified=True,
        paper_composition_alpha=0.45,
        scoring_changes="scoring_changes.json",
        evidence_files=verification["evidence_files"],
        inputs_unchanged=True,
        small_molecule_cases=5,
        H2O_symmetry_passed=True,
        fresh_SCF_calculations=6,
        new_OpenMolcas_Block2_jobs=0,
        full_dio_TiF3_solver_rerun=False,
        python=platform.python_version(),
        numpy=numpy.__version__,
        scipy=scipy.__version__,
        h5py=h5py.__version__,
        installed_package=str(PACKAGE),
        evidence_root=str(evidence.root),
        elapsed_seconds=time.monotonic() - started,
    )
    write_new(output / "summary.json", report)
    print(f"PAS {RELEASE} acceptance PASSED: " + str(output / "summary.json"), flush=True)
    return report
