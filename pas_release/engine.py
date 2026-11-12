"""Boundary for replay and release adapters using the scientific engine."""

from contextlib import contextmanager
from unittest.mock import patch

from pas import backend, controller


def budgets(manifest):
    return controller._run_budgets(manifest)


def fixed_policy(manifest, number):
    return controller._fixed_dmrg_policy(manifest, number)


def belief(context, config, receipts, seed, bootstrap):
    return controller._belief(context, config, receipts, seed, bootstrap)


def input_domain(context, bootstrap, receipts, **options):
    return controller._input_domain(context, bootstrap, receipts, **options)


def optimized(context, receipt):
    return controller._optimized_output(context, receipt)


def verify_receipt(receipt, file_hash):
    with patch.object(controller, "file_hash", file_hash):
        controller._verify_receipt(receipt)


def verify_run(directory):
    from pathlib import Path

    from pas.common import read_json

    directory = Path(directory)
    manifest = controller._verify_manifest(directory)
    state = read_json(directory / "state.json")
    seed = read_json(directory / "seed.json")
    controller._verify_state(directory, state, manifest, seed)
    return manifest, state


def validate_policies(config, policies):
    from pas.config import planning_configuration, validate_production_plan
    from pas.diagnostic_window import validate_policy
    from pas.planner import fixed_planner_policy

    planning_configuration(config, policies.get("response_policy"))
    planner = fixed_planner_policy(policies.get("planner_policy"))
    controller._verify_survey_policy(config, planner)
    if policies.get("survey_window_policy") is not None:
        validate_policy(policies["survey_window_policy"])
    if policies.get("fixed_dmrg_policy") is not None:
        controller._verify_fixed_dmrg_transition({"config": config, **policies}, 0)
    budgets({"config": config, **policies})
    validate_production_plan(config)


def forbidden(*args, **kwargs):
    raise RuntimeError("Physical execution and ledger writes are prohibited during replay")


@contextmanager
def read_only():
    with (
        patch.object(backend, "execute", side_effect=forbidden),
        patch.object(controller, "write_json", side_effect=forbidden),
    ):
        yield
