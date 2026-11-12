"""The supported PAS command line: explicit calculations and read-only checks."""

import argparse
import json
import os
import shlex
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from . import __version__
from .common import require, write_new


def parser():
    root = argparse.ArgumentParser(
        prog="pas", description="PAS paper program with the c06614a numerical baseline"
    )
    root.add_argument("--version", action="version", version="PAS " + __version__)
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Create an editable input; no calculation is started")
    init.add_argument("profile", nargs="?", help="A paper example, cas, or dmrg; see --list")
    init.add_argument(
        "--list", action="store_true", dest="list_profiles", help="List available input templates"
    )
    init.add_argument(
        "--output", type=Path, help="New input JSON file; existing files are never overwritten"
    )
    source = init.add_mutually_exclusive_group()
    source.add_argument(
        "--seed-file",
        type=Path,
        help="Use an existing PAS seed (path relative to this working directory)",
    )
    source.add_argument(
        "--case-file",
        type=Path,
        help="Use a molecular JSON file (path relative to this working directory)",
    )
    check = commands.add_parser(
        "check", help="Validate an input and report native runtime readiness without computing"
    )
    check.add_argument("input", type=Path)
    run = commands.add_parser(
        "run", help="Start a new native calculation under the original cu07 constraints"
    )
    run.add_argument("input", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument(
        "--prepare-only",
        action="store_true",
        help="Import a complete seed and prepare a ledger without solver execution",
    )
    status = commands.add_parser(
        "status", help="Read and verify an existing run without modifying it"
    )
    status.add_argument("directory", type=Path)
    resume = commands.add_parser(
        "resume", help="Continue an existing immutable run; changed inputs require a new run"
    )
    resume.add_argument("directory", type=Path)
    reproduce = commands.add_parser(
        "reproduce", help="Validate the paper evidence or recompute the small-molecule controls"
    )
    reproduce.add_argument("target", choices=("paper", "small-molecules"))
    reproduce.add_argument(
        "--evidence", type=Path, help="Extracted evidence/ directory, required for paper replay"
    )
    reproduce.add_argument("--output", type=Path, required=True, help="A new result directory")
    verify = commands.add_parser(
        "verify", help="Verify installed scientific code and frozen reference material"
    )
    verify.add_argument("--evidence", type=Path)
    offline = commands.add_parser(
        "offline", help="Analyze external fixed-orbital data without a quantum-chemistry solver"
    )
    actions = offline.add_subparsers(dest="offline_action", required=True)
    offline_example = actions.add_parser("example", help="Copy the complete reviewer example")
    offline_example.add_argument("--output", type=Path, required=True)
    offline_check = actions.add_parser(
        "check", help="Validate external data and physical conventions"
    )
    offline_check.add_argument("input", type=Path)
    offline_select = actions.add_parser(
        "select", help="Recommend an active space from imported observations"
    )
    offline_select.add_argument("input", type=Path)
    offline_select.add_argument("--output", type=Path, required=True)
    preflight = commands.add_parser(
        "preflight", help="Inspect native cluster requirements without creating a run"
    )
    preflight.add_argument("--output-directory", type=Path)
    preflight.add_argument("--threads", type=int, default=32)
    preflight.add_argument("--backend", choices=("cas", "dmrg"), default="dmrg")
    for command in (
        init,
        check,
        run,
        status,
        resume,
        reproduce,
        verify,
        preflight,
        offline_example,
        offline_check,
        offline_select,
    ):
        command.add_argument(
            "--json",
            action="store_true",
            help="Write a single machine-readable JSON result to stdout",
        )
    return root


def _emit(result, as_json):
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    elif result.get("mode", "").startswith("offline_"):
        print(result["summary"], end="" if result["summary"].endswith("\n") else "\n")
    elif "templates" in result:
        print("Input templates:")
        for name, entry in result["templates"].items():
            print(f"  {name:20} {entry['description']}")
        print("Create: pas init PROFILE --output input.json")
        print("Paper templates use original cu07 seeds; cas/dmrg create editable molecular inputs.")
    elif "created_files" in result:
        for path in result["created_files"]:
            print("Created: " + path)
        if result["molecule_file"]:
            print("Starter: H2 / STO-3G / 2 electrons in 2 orbitals, two singlet roots.")
            print(
                "Edit the molecular file: geometry, basis, charge, active space, states and AO target."
            )
        else:
            print("Initialization: " + next(iter(result["initialization"].values())))
        print("Check before running: pas check " + shlex.quote(result["input_file"]))
        print("Template creation does not validate physical files or start a calculation.")
    elif "mode" in result:
        from .outputs import summary_text

        print(summary_text(result), end="")
    elif "input_valid" in result:
        system, config = result["system"], result["config"]
        print("Input: valid — " + result["input_file"])
        print("Name: " + result["name"])
        print("Profile: " + result["profile"])
        print(
            f"Method: {config['backend'].upper()}; active space: {system['active_electrons']} electrons / {system['active_orbitals']} orbitals"
        )
        print(
            f"Spin multiplicity: {system['spin_multiplicity']}; roots: {system['roots']}; weights: {system['weights']}"
        )
        print(
            "Evidence (relative weights): "
            + ", ".join(
                f"{channel}={config['weights'][channel]:g}" for channel in config["channels"]
            )
        )
        print("Evidence workflow: " + config.get("survey_model", "default"))
        print(f"Composition alpha: {config.get('composition_purity_weight', 0.5):g}")
        print(
            f"Threads: {config['threads']}; budgets: {config['diagnostic_budget']} diagnostics / {config['optimization_budget']} optimizations"
        )
        print(f"Fixed random seed: {config['seed']}; settings source: {result['settings_source']}")
        initialization_labels = {
            "frozen_paper_seed": "reuse the unchanged paper initialization",
            "imported_seed": "import an existing PAS initialization",
            "cold_start": "initialize from the molecular input",
        }
        print("Initialization: " + initialization_labels[result["initialization_mode"]])
        print(
            "Settings changed from paper profile: " + (", ".join(result["changed_settings"]) or "none")
        )
        if result["historical_settings_changes"]:
            print(
                "Settings differing from historical receipts: "
                + ", ".join(result["historical_settings_changes"])
            )
        print(
            "Native runtime: " + ("ready" if result["can_run_here"] else "unavailable on this host")
        )
        if not result["can_run_here"]:
            environment = result["environment"]
            if not environment["checks"]["host_is_cu07"]:
                print("  Run native calculations on cu07; current host: " + environment["host"])
            else:
                for key, ok in environment["checks"].items():
                    if not ok:
                        print(
                            "  Check "
                            + key
                            + "; run 'pas preflight --json' for required paths and work-root details"
                        )
    else:
        print("PASS" if result["passed"] else "NOT READY")
        for key, value in result.get("checks", {}).items():
            print(f"  {key}: {'ok' if value else 'unavailable'}")


def _reproduce(args):
    from . import verify
    from .outputs import replay_results, small_molecule_results, write_results
    from .resources import PACKAGE

    output = args.output.resolve()
    require(
        not output.is_relative_to(PACKAGE), "Result directory must be outside the installed package"
    )
    if args.target == "paper":
        require(args.evidence is not None, "--evidence is required for paper acceptance")
        require(
            not output.is_relative_to(args.evidence.resolve()),
            "Output must be outside the immutable evidence tree",
        )
    require(not output.exists(), "Output already exists; choose a new directory")
    verify.run(args.evidence if args.target == "paper" else None)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "run.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                if args.target == "paper":
                    from .reproduction.replay import run

                    details = run(args.evidence, output / "validation")
                    cases, history = replay_results(args.evidence, output / "validation")
                else:
                    from .reproduction.small_molecules import run

                    details = run()
                    write_new(output / "validation/small_molecules.json", details)
                    cases, history = small_molecule_results(details), []
                result = dict(
                    schema="pas.result.v1",
                    mode="replay",
                    name=args.target,
                    status="passed",
                    passed=True,
                    orbital_index_base=1,
                    energy_unit="hartree",
                    cases=cases,
                    details=details,
                    reports="validation/",
                )
                write_results(output, result, history)
                return result
            except Exception as error:
                traceback.print_exc(file=log)
                write_results(
                    output,
                    dict(
                        schema="pas.result.v1",
                        mode="failure",
                        name=args.target,
                        status="failed",
                        passed=False,
                        error=str(error),
                        reports="validation/",
                    ),
                )
                raise


def _error(error, code, as_json):
    if as_json:
        print(
            json.dumps(dict(error=str(error), exit_code=code), ensure_ascii=False), file=sys.stderr
        )
    else:
        print("pas: " + str(error), file=sys.stderr)
    return code


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command in ("reproduce", "verify", "offline"):
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = "1"
    from .cluster import EnvironmentError, preflight

    try:
        code = 0
        if args.command == "init":
            from .templates import catalog, create

            if args.list_profiles or (
                args.profile is None
                and args.output is None
                and args.seed_file is None
                and args.case_file is None
            ):
                require(
                    args.profile is None
                    and args.output is None
                    and args.seed_file is None
                    and args.case_file is None,
                    "Use 'pas init --list' alone, or choose a profile with --output",
                )
                result = dict(templates=catalog())
            else:
                require(
                    args.profile is not None and args.output is not None,
                    "Choose a template and destination: pas init PROFILE --output input.json; see 'pas init --list'",
                )
                result = create(
                    args.profile, args.output, seed_file=args.seed_file, case_file=args.case_file
                )
        elif args.command == "check":
            from .common import read_json, verify_source
            from .inputs import load_input
            from .profiles import paper_configuration
            from .resources import DATA, PROFILE_NAMES

            verify_source()
            spec = load_input(args.input)
            environment = preflight(
                threads=spec["config"]["threads"], backend_name=spec["config"]["backend"]
            )
            case = read_json(DATA / "cases.json")["cases"][PROFILE_NAMES[spec["profile"]]]
            source_config = paper_configuration(case)
            result = dict(
                input_valid=True,
                input_file=spec["input_file"]["path"],
                name=spec["name"],
                profile=spec["profile"],
                initialization_mode=spec["initialization_mode"],
                settings_source=PROFILE_NAMES[spec["profile"]],
                changed_settings=[
                    key
                    for key in sorted(spec["config"].keys() | source_config.keys())
                    if spec["config"].get(key) != source_config.get(key)
                ],
                historical_settings_changes=[
                    key
                    for key in sorted(spec["config"].keys() | case["config"].keys())
                    if spec["config"].get(key) != case["config"].get(key)
                ],
                paper_profile_unchanged=spec["paper_profile_unchanged"],
                historical_profile_unchanged=spec["historical_profile_unchanged"],
                companion_inputs={
                    key: value
                    for key, value in spec["files"].items()
                    if key in ("references_file", "bootstrap_evidence_file")
                },
                config_id=spec["config_id"],
                can_run_here=environment["passed"],
                system=spec["system"],
                config=spec["config"],
                policies=spec["policies"],
                environment=environment,
            )
        elif args.command in ("run", "resume", "status"):
            from . import runtime

            if args.command == "run":
                result = runtime.start(args.input, args.output, prepare_only=args.prepare_only)
            else:
                result = getattr(runtime, args.command)(args.directory)
        elif args.command == "offline":
            from . import offline

            if args.offline_action == "example":
                result = offline.example(args.output)
            elif args.offline_action == "check":
                result = offline.check(args.input)
            else:
                result = offline.select(args.input, args.output)
        elif args.command == "reproduce":
            result = _reproduce(args)
        elif args.command == "verify":
            from .verify import run

            result = run(args.evidence)
        else:
            result = preflight(args.output_directory, args.threads, args.backend)
            code = 0 if result["passed"] else 3
        _emit(result, args.json)
        return code
    except (ValueError, KeyError, TypeError, FileNotFoundError) as error:
        return _error(error, 2, args.json)
    except EnvironmentError as error:
        return _error(error, 3, args.json)
    except KeyboardInterrupt:
        print("pas: interrupted; inspect the existing ledger before resuming", file=sys.stderr)
        return 130
    except Exception as error:
        return _error(error, 1, args.json)


if __name__ == "__main__":
    raise SystemExit(main())
