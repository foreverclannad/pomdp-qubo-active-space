"""One controller for all evidence masks and solver backends.

References are visible only to the external stop callback. An interrupted request
is reconciled from its committed receipt, never silently executed a second time.
"""

from __future__ import annotations

import os
import time
import traceback
from copy import deepcopy
from pathlib import Path

import numpy as np

from .cluster import runtime_bindings
from .common import file_hash, identity, read_json, write_json
from .config import configuration, planning_configuration, validate_production_plan
from .contracts import Context
from .diagnostic_window import (
    derive_window,
    survey_partition,
)
from .diagnostic_window import (
    validate_policy as validate_window_policy,
)


def source_files():
    """Hash the installed scientific implementation with portable path names."""
    root = Path(__file__).resolve().parent
    return {
        p.relative_to(root).as_posix(): file_hash(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".c")
    }


def source_identity():
    return identity(source_files())


def _run_budgets(manifest):
    """Execution limits may increase without relabelling physical evidence."""
    config = configuration(manifest["config"])
    limits = {key: config[key] for key in ("diagnostic_budget", "optimization_budget")}
    if "budget_policy" not in manifest:
        return limits
    policy = manifest["budget_policy"]
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema", *limits}
        or policy.get("schema") != "pas_run_budget_v1"
        or any(type(policy[key]) is not int or policy[key] < value for key, value in limits.items())
    ):
        raise ValueError(
            "Run budget policy must explicitly preserve or increase both integer limits"
        )
    limits = {key: policy[key] for key in limits}
    if config["backend"] in ("cas", "dmrg"):
        validate_production_plan({**config, **limits})
    return limits


def _backend(config):
    if config["backend"] == "pyscf":
        from . import pyscf_backend

        return pyscf_backend
    from . import backend

    return backend


def _requires_context_survey(config):
    return config.get("survey_model") == "posterior_evidence_capture_v1"


def _verify_survey_policy(config, policy):
    if _requires_context_survey(config) and (
        not config["full_window_survey"]
        or policy != dict(schema="pas_optimize_only_two_step_v1", depth=2)
    ):
        raise ValueError(
            "Posterior evidence capture requires whole-window surveys and optimize-only H2"
        )


def _fixed_dmrg_policy(manifest, action_number):
    """Resolve the immutable protocol boundary without changing any Context."""
    transition = manifest.get("fixed_dmrg_policy")
    if transition is None or action_number < transition["from_action"]:
        return None
    return deepcopy(transition["policy"])


def _verify_fixed_dmrg_transition(manifest, completed_actions):
    if "fixed_dmrg_policy" not in manifest:
        return
    from .solvers.block2_convergence import validate_fixed_policy

    transition = manifest["fixed_dmrg_policy"]
    if (
        not isinstance(transition, dict)
        or set(transition) != {"schema", "from_action", "policy"}
        or transition["schema"] != "pas_fixed_dmrg_transition_v1"
        or type(transition["from_action"]) is not int
        or not 1 <= transition["from_action"] <= completed_actions + 1
        or manifest["config"]["backend"] != "dmrg"
        or transition["policy"] is None
    ):
        raise ValueError("Invalid fixed DMRG protocol transition or action boundary")
    validate_fixed_policy(transition["policy"])


def _survey_key(context_id, fixed_dmrg_policy):
    return context_id, identity(fixed_dmrg_policy)


def _verify_survey_action(
    kind, context_id, attempts, *, valid=False, bootstrap=False, fixed_dmrg_policy=None
):
    """One durable survey attempt per context and physical protocol."""
    key = _survey_key(context_id, fixed_dmrg_policy)
    if kind == "probe":
        raise ValueError("Posterior evidence capture prohibits target-CAS probes")
    if kind == "survey":
        if bootstrap:
            raise ValueError("Survey cannot replace frozen bootstrap input in its bound context")
        if key in attempts:
            raise ValueError(
                "A whole-window survey was already attempted in this context; automatic retry prohibited"
            )
        attempts[key] = bool(valid)
    elif kind == "optimize" and not (attempts.get(key, False) or bootstrap):
        raise ValueError("Optimization requires a valid whole-window survey in its input context")


def _survey_posterior(receipt, context, config):
    if not (
        _requires_context_survey(config) and receipt["kind"] == "survey" and receipt.get("valid")
    ):
        return None
    fields = receipt.get("proposal_quadratic", {})
    posterior = fields.get("input_posterior", {})
    partition = survey_partition(context, receipt["active"])
    if (
        fields.get("score_model") != "posterior_evidence_capture_v1"
        or posterior.get("schema") != "pas_capture_statistics_posterior_v1"
        or posterior.get("context_id") != context.context_id
        or posterior.get("basis_id") != context.basis_id
        or fields.get("measured_indices") != partition["active"]
        or posterior.get("measured_indices") != partition["active"]
        or posterior.get("electrons") != partition["electrons"]
        or not posterior.get("raw_evidence_hash")
        or not posterior.get("model_hash")
        or isinstance(posterior.get("gain"), bool)
        or not isinstance(posterior.get("gain"), (int, float))
        or not np.isfinite(posterior["gain"])
        or not 0 <= posterior["gain"] <= 1
    ):
        raise ValueError("Valid survey lacks its bound input-evidence posterior")
    return posterior


def prepare(
    seed,
    config,
    directory,
    references=None,
    *,
    planner_policy=None,
    bootstrap_input_evidence=None,
    response_policy=None,
    survey_window_policy=None,
    fixed_dmrg_policy=None,
    budget_policy=None,
):
    import fcntl

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "preparation.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _prepare(
            seed,
            config,
            directory,
            references,
            planner_policy=planner_policy,
            bootstrap_input_evidence=bootstrap_input_evidence,
            response_policy=response_policy,
            survey_window_policy=survey_window_policy,
            fixed_dmrg_policy=fixed_dmrg_policy,
            budget_policy=budget_policy,
        )


def _prepare(
    seed,
    config,
    directory,
    references=None,
    *,
    planner_policy=None,
    bootstrap_input_evidence=None,
    response_policy=None,
    survey_window_policy=None,
    fixed_dmrg_policy=None,
    budget_policy=None,
):
    from .planner import fixed_planner_policy

    planner_policy = fixed_planner_policy(planner_policy)
    directory = Path(directory).resolve()
    state_path = directory / "state.json"
    if state_path.exists():
        raise FileExistsError("A prepared run must be resumed, not overwritten")
    config = configuration(config)
    if fixed_dmrg_policy is not None:
        _verify_fixed_dmrg_transition(dict(config=config, fixed_dmrg_policy=fixed_dmrg_policy), 0)
    planning_config = planning_configuration(config, response_policy)
    _verify_survey_policy(config, planner_policy)
    if survey_window_policy is not None:
        validate_window_policy(survey_window_policy)
        if not _requires_context_survey(config):
            raise ValueError("A diagnostic window policy requires per-context posterior surveys")
    if config["backend"] in ("cas", "dmrg"):
        validate_production_plan(config)
    context = Context.from_dict(seed["context"])
    if config["backend"] in ("cas", "dmrg") and not context.electron_roles:
        raise ValueError(
            "Native runs require SCF-derived electron roles; re-import the initialization"
        )
    if config["backend"] in ("cas", "dmrg"):
        from .electron_roles import verify_context

        verify_context(context)
    incumbent = seed["incumbent"]
    if identity(incumbent.get("output_context")) != identity(context.to_dict()):
        raise ValueError("Initial incumbent belongs to a different physical context")
    if context.backend != config["backend"]:
        raise ValueError("Initial backend differs from the requested backend")
    if (
        "composition" not in config["channels"]
        and seed.get("initialization", {}).get("initial_selection_used_composition") is not False
    ):
        raise ValueError("Composition-independent initialization provenance is required")
    if context.effective_config_id != identity(config):
        raise ValueError("Initial context does not bind the complete effective run configuration")
    _verify_receipt(incumbent)
    _validate_metrics(incumbent, config)
    if not all(incumbent.get(k) for k in ("valid", "converged", "state_valid")):
        raise ValueError("A valid physical cold start is required")
    if config.get("response_model") == "shared_projectors_v1":
        # Validate the common AO descriptor before reserving any physical work.
        from .belief import ResponseBelief

        ResponseBelief(context, planning_config)
    reference_binding = None
    if references is not None:
        references = Path(references).resolve()
        reference_binding = dict(path=str(references), sha256=file_hash(references))
    manifest = dict(
        schema="pas_run_v1",
        source_id=source_identity(),
        config=config,
        runtime=runtime_bindings(config),
        seed_id=identity(seed),
        references=reference_binding,
        planner_policy=planner_policy,
    )
    if response_policy is not None:
        manifest["response_policy"] = dict(response_policy)
    if survey_window_policy is not None:
        manifest["survey_window_policy"] = dict(survey_window_policy)
    if fixed_dmrg_policy is not None:
        manifest["fixed_dmrg_policy"] = deepcopy(fixed_dmrg_policy)
    if budget_policy is not None:
        manifest["budget_policy"] = deepcopy(budget_policy)
    budgets = _run_budgets(manifest)
    if bootstrap_input_evidence is not None:
        from .input_evidence import bootstrap_input

        record = bootstrap_input(bootstrap_input_evidence, context, config)
        path = directory / "bootstrap_input_evidence.json"
        write_json(path, record)
        manifest["bootstrap_input_evidence"] = dict(path=str(path), sha256=file_hash(path))
    manifest["manifest_id"] = identity(manifest)
    state = dict(
        schema="pas_state_v1",
        manifest_id=manifest["manifest_id"],
        status="prepared",
        context=context.to_dict(),
        incumbent=incumbent,
        initialization=seed.get("initialization", {}),
        remaining_diagnostics=budgets["diagnostic_budget"],
        remaining_optimizations=budgets["optimization_budget"],
        survey_collected=False,
        pending=None,
        events=[],
        contexts=[context.context_id],
        search_cpu_seconds=0.0,
        search_wall_seconds=0.0,
        planning_wall_seconds=0.0,
        created_unix=time.time(),
    )
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "seed.json", seed)
    write_json(directory / "manifest.json", manifest)
    write_json(state_path, state)
    return state


def _validate_metrics(receipt, config):
    if not receipt.get("valid"):
        return
    metrics = receipt.get("metrics", {})
    if not all(k in metrics for k in (*config["channels"], "quality")):
        raise ValueError("A valid result is missing an enabled terminal metric")
    values = np.array([metrics[k] for k in config["channels"]], float)
    if not np.all(np.isfinite(values)) or np.any(values < -1e-7) or np.any(values > 1 + 1e-7):
        raise ValueError("Physical quality metric outside its declared range")
    weights = np.array([config["weights"][k] for k in config["channels"]])
    if not np.isclose(metrics["quality"], weights @ values / weights.sum(), atol=1e-12, rtol=0):
        raise ValueError("Quality disagrees with the frozen enabled-channel weights")


def _verify_receipt(receipt):
    if receipt.get("evidence_id") != identity(
        {k: v for k, v in receipt.items() if k != "evidence_id"}
    ):
        raise ValueError("Evidence identity/content conflict")
    for item in receipt.get("artifacts", {}).values():
        if isinstance(item, dict) and {"path", "sha256"} <= item.keys():
            if file_hash(item["path"]) != item["sha256"]:
                raise ValueError("Receipt physical artifact changed")


def _verify_request(action_directory, context, receipt, config, *, fixed_dmrg_policy=None):
    """A completed action is evidence only for its immutable physical request."""
    request_file = Path(action_directory).resolve() / "request.json"
    if not request_file.is_file():
        raise ValueError("Completed physical receipt is missing its mandatory request.json")
    request = read_json(request_file)
    if not isinstance(request, dict) or request.get("request_id") != identity(
        {k: v for k, v in request.items() if k != "request_id"}
    ):
        raise ValueError("Physical request identity/content conflict")
    if request["request_id"] != receipt["request_id"]:
        raise ValueError("Receipt request identity differs from the physical dispatch")
    for record in (request, receipt):
        if fixed_dmrg_policy is None:
            if "fixed_dmrg_policy" in record:
                raise ValueError("Legacy physical request/receipt cannot claim a fixed DMRG policy")
        elif "fixed_dmrg_policy" not in record or record["fixed_dmrg_policy"] != fixed_dmrg_policy:
            raise ValueError(
                "Physical request/receipt fixed DMRG policy differs from its action boundary"
            )
    kind = receipt["kind"]
    if kind == "survey":
        if context.backend != "dmrg" and not (
            context.backend == "cas" and _requires_context_survey(config)
        ):
            raise ValueError("Survey request requires DMRG or explicit passive CAS evidence")
        # Validate the actual frozen physical window, not a later window policy.
        partition = survey_partition(context, receipt["active"])
    elif kind in ("probe", "optimize"):
        partition = context.partition(receipt["active"])
    else:
        raise ValueError("Completed request has an unsupported action kind")
    expected = dict(
        schema="pas_physical_request_v1",
        context_id=context.context_id,
        basis_id=context.basis_id,
        checkpoint_hash=context.payload.get("checkpoint_hash", context.basis_id),
        kind=kind,
        partition=partition,
        protocol=config,
    )
    for key, value in expected.items():
        if key not in request or identity(request[key]) != identity(value):
            raise ValueError(
                "Physical request " + key + " differs from the frozen input context/protocol"
            )
    if not isinstance(request.get("input"), str) or not request["input"].strip():
        raise ValueError("Physical request lacks its effective solver input")
    artifact = receipt.get("artifacts", {}).get("request")
    if (
        not isinstance(artifact, dict)
        or not {"path", "sha256"} <= artifact.keys()
        or Path(artifact["path"]).resolve() != request_file
        or artifact["sha256"] != file_hash(request_file)
    ):
        raise ValueError("Receipt must bind the exact physical request artifact")
    if kind == "survey" and context.backend == "cas":
        from .backend import verify_cas_survey

        verify_cas_survey(context, partition, config, request, receipt)
    return request


def _optimized_output(context, receipt):
    if receipt.get("valid") and not all(receipt.get(k) for k in ("converged", "state_valid")):
        raise ValueError("Valid receipt contradicts convergence/state checks")
    if receipt["kind"] != "optimize" or not receipt.get("valid"):
        return None
    output = Context.from_dict(receipt["output_context"])
    fields = (
        "geometry_id",
        "backend",
        "nmo",
        "total_electrons",
        "electrons",
        "k",
        "states",
        "protocol_id",
        "effective_config_id",
        "required_active",
    )
    if any(getattr(output, key) != getattr(context, key) for key in fields):
        raise ValueError("Optimized context changed the physical problem")
    if len(output.frozen) != len(context.frozen) or len(output.deleted) != len(context.deleted):
        raise ValueError("Optimized context changed frozen/deleted dimensions")
    if sorted(output.electron_roles) != sorted(context.electron_roles):
        raise ValueError("Optimized context changed the reference electron-role inventory")
    if context.backend in ("cas", "dmrg"):
        from .electron_roles import verify_context

        verify_context(output)
    return output


def _verify_manifest(directory):
    manifest = read_json(directory / "manifest.json")
    _run_budgets(manifest)
    if manifest["config"]["backend"] in ("cas", "dmrg"):
        validate_production_plan(manifest["config"])
    if manifest["manifest_id"] != identity(
        {k: v for k, v in manifest.items() if k != "manifest_id"}
    ):
        raise ValueError("Run manifest changed")
    if manifest["source_id"] != source_identity():
        raise ValueError("Frozen source changed; create a new run rather than mixing versions")
    _verify_fixed_dmrg_transition(manifest, len(read_json(directory / "state.json")["events"]))
    from .planner import fixed_planner_policy

    # Absent policy uses this source version's explicit default. It never waives
    # the source identity guard above or changes the physical request protocol.
    if "planner_policy" in manifest and manifest["planner_policy"] is None:
        raise ValueError("Explicit planner policy must have its schema and ratio")
    policy = fixed_planner_policy(manifest.get("planner_policy"))
    _verify_survey_policy(manifest["config"], policy)
    if "response_policy" in manifest and manifest["response_policy"] is None:
        raise ValueError("Explicit response policy must have its schema and model")
    planning_configuration(manifest["config"], manifest.get("response_policy"))
    if "survey_window_policy" in manifest:
        validate_window_policy(manifest["survey_window_policy"])
        if not _requires_context_survey(manifest["config"]):
            raise ValueError("A diagnostic window policy requires per-context posterior surveys")
    if manifest.get("runtime") != runtime_bindings(manifest["config"]):
        raise ValueError(
            "Frozen solver executables or numerical libraries changed; physical dispatch is prohibited"
        )
    if identity(read_json(directory / "seed.json")) != manifest["seed_id"]:
        raise ValueError("Frozen cold start changed")
    if "budget_transition" in manifest:
        bound = manifest["budget_transition"]
        if file_hash(bound["path"]) != bound["sha256"]:
            raise ValueError("Frozen budget transition changed")
        record = read_json(bound["path"])
        if (
            record.get("schema") != "pas_run_budget_transition_v1"
            or record.get("new_source_id") != manifest["source_id"]
            or record.get("budget_policy") != manifest.get("budget_policy")
            or record.get("physical_config_id") != identity(manifest["config"])
            or record.get("seed_id") != manifest["seed_id"]
            or record.get("previous_manifest") != manifest.get("previous_manifest")
        ):
            raise ValueError("Budget transition differs from the continued manifest")
        previous = record["previous_manifest"]
        if file_hash(previous["path"]) != previous["sha256"]:
            raise ValueError("Original budget manifest backup changed")
    bootstrap = _bootstrap_input(directory, manifest)
    if bootstrap is not None:
        from .input_evidence import bootstrap_input

        context = Context.from_dict(read_json(directory / "seed.json")["context"])
        replayed = bootstrap_input(bootstrap["source"]["path"], context, manifest["config"])
        if identity(replayed) != identity(bootstrap):
            raise ValueError("Frozen bootstrap input evidence changed")
    binding = manifest.get("references")
    if binding and file_hash(binding["path"]) != binding["sha256"]:
        raise ValueError("External reference manifest changed")
    return manifest


def _bootstrap_input(directory, manifest):
    binding = manifest.get("bootstrap_input_evidence")
    if binding is None:
        return None
    path = Path(binding["path"]).resolve()
    if (
        path != (Path(directory) / "bootstrap_input_evidence.json").resolve()
        or file_hash(path) != binding["sha256"]
    ):
        raise ValueError("Frozen bootstrap input binding changed")
    record = read_json(path)
    if record.get("schema") != "pas_bootstrap_input_evidence_v1":
        raise ValueError("Unsupported bootstrap input schema")
    return record


def _bootstrap_domain(context, bootstrap):
    if bootstrap is not None and bootstrap["context_id"] == context.context_id:
        from .input_evidence import MeasuredDomain

        return MeasuredDomain(context, bootstrap["measured_window"])
    return context


def _input_domain(context, bootstrap, receipts, *, fixed_dmrg_policy=None):
    """Only measured orbitals can be proposed; physical context identity stays intact."""
    survey = next(
        (
            r
            for r in reversed(receipts)
            if r["kind"] == "survey"
            and r.get("valid")
            and r["context_id"] == context.context_id
            and r.get("fixed_dmrg_policy") == fixed_dmrg_policy
        ),
        None,
    )
    if survey is not None:
        from .input_evidence import MeasuredDomain

        return MeasuredDomain(context, survey["active"]), survey, False
    domain = _bootstrap_domain(context, bootstrap)
    return domain, bootstrap if domain is not context else None, domain is not context


def _read_events(directory, state):
    result = []
    for event in state["events"]:
        path = directory / event["receipt"]
        if (
            event["receipt"] != f"actions/{event['number']:03d}_{event['kind']}/receipt.json"
            or not path.resolve().is_relative_to(directory.resolve())
        ):
            raise ValueError("Committed receipt path differs from its reserved action")
        if file_hash(path) != event["sha256"]:
            raise ValueError("Committed evidence changed")
        receipt = read_json(path)
        _verify_receipt(receipt)
        result.append(receipt)
    return result


def _verify_state(directory, state, manifest, seed):
    """Reconcile budgets, incumbent and costs with the durable evidence ledger."""
    if state["manifest_id"] != manifest["manifest_id"]:
        raise ValueError("State belongs to a different manifest")
    cfg = manifest["config"]
    budgets = _run_budgets(manifest)
    _verify_fixed_dmrg_transition(manifest, len(state["events"]))
    bootstrap = _bootstrap_input(directory, manifest)
    receipts = _read_events(directory, state)
    incumbent, context = seed["incumbent"], Context.from_dict(seed["context"])
    if cfg["backend"] in ("cas", "dmrg") and not context.electron_roles:
        raise ValueError("Native runs require SCF-derived electron roles")
    if cfg["backend"] in ("cas", "dmrg"):
        from .electron_roles import verify_context

        verify_context(context)
    if context.effective_config_id != identity(cfg):
        raise ValueError("Initial context no longer binds the complete effective run configuration")
    spent = dict(probe=0, survey=0, optimize=0)
    survey_attempts = {}
    ids, requests = set(), set()
    costs = dict(cpu_seconds=0.0, wall_seconds=0.0)
    for number, (event, receipt) in enumerate(zip(state["events"], receipts), 1):
        fixed_policy = _fixed_dmrg_policy(manifest, number)
        kind = receipt["kind"]
        if (
            kind not in spent
            or event["number"] != number
            or event["kind"] != kind
            or event["evidence_id"] != receipt["evidence_id"]
            or event["context_id"] != context.context_id
            or receipt["context_id"] != context.context_id
            or bool(event["valid"]) != bool(receipt["valid"])
        ):
            raise ValueError("Event ledger order, context or receipt binding differs")
        if receipt["evidence_id"] in ids or receipt["request_id"] in requests:
            raise ValueError("The event ledger consumed the same request more than once")
        if _requires_context_survey(cfg):
            if spent["optimize"] >= budgets["optimization_budget"]:
                raise ValueError("Physical action recorded after optimization budget exhaustion")
            domain, _, bootstrap_active = _input_domain(
                context, bootstrap, receipts[: number - 1], fixed_dmrg_policy=fixed_policy
            )
            _verify_survey_action(
                kind,
                context.context_id,
                survey_attempts,
                valid=receipt["valid"],
                bootstrap=bootstrap_active,
                fixed_dmrg_policy=fixed_policy,
            )
            if kind == "optimize":
                domain.partition(receipt["active"])
        _verify_request(
            (directory / event["receipt"]).parent,
            context,
            receipt,
            cfg,
            fixed_dmrg_policy=fixed_policy,
        )
        _survey_posterior(receipt, context, cfg)
        output = _optimized_output(context, receipt)
        ids.add(receipt["evidence_id"])
        requests.add(receipt["request_id"])
        spent[kind] += 1
        if kind != "survey":
            _validate_metrics(receipt, cfg)
        accepted = bool(
            kind == "optimize"
            and receipt["valid"]
            and receipt["metrics"]["quality"]
            > incumbent["metrics"]["quality"] + cfg["acceptance_tolerance"]
        )
        if event["accepted"] != accepted:
            raise ValueError("Accepted incumbent history violates the frozen quality rule")
        if accepted:
            incumbent, context = receipt, output
        for key in costs:
            value = float(receipt[key])
            if not np.isfinite(value) or value < 0:
                raise ValueError("Invalid cost in evidence ledger")
            costs[key] += value
    pending = state["pending"]
    if pending:
        fixed_policy = _fixed_dmrg_policy(manifest, len(receipts) + 1)
        if (
            pending.get("fixed_dmrg_policy") != fixed_policy
            or fixed_policy is None
            and "fixed_dmrg_policy" in pending
        ):
            raise ValueError("Pending request fixed DMRG policy differs from its action boundary")
        if (
            pending["number"] != len(receipts) + 1
            or pending["context_id"] != context.context_id
            or pending["kind"] not in spent
        ):
            raise ValueError("Pending request differs from the evidence ledger")
        if pending["directory"] != f"actions/{pending['number']:03d}_{pending['kind']}":
            raise ValueError("Pending request directory differs from its reserved action")
        if _requires_context_survey(cfg):
            if spent["optimize"] >= budgets["optimization_budget"]:
                raise ValueError("Physical action reserved after optimization budget exhaustion")
            domain, _, bootstrap_active = _input_domain(
                context, bootstrap, receipts, fixed_dmrg_policy=fixed_policy
            )
            _verify_survey_action(
                pending["kind"],
                context.context_id,
                survey_attempts,
                bootstrap=bootstrap_active,
                fixed_dmrg_policy=fixed_policy,
            )
            if pending["kind"] == "optimize":
                domain.partition(pending["active"])
        spent[pending["kind"]] += 1
    if (not _requires_context_survey(cfg) and spent["survey"] > 1) or state[
        "survey_collected"
    ] != any(r["kind"] == "survey" for r in receipts):
        raise ValueError("Whole-window survey must be collected once in the initial context")
    if (
        state["remaining_diagnostics"]
        != budgets["diagnostic_budget"] - spent["probe"] - spent["survey"]
        or state["remaining_optimizations"] != budgets["optimization_budget"] - spent["optimize"]
        or min(state["remaining_diagnostics"], state["remaining_optimizations"]) < 0
    ):
        raise ValueError("Remaining budget differs from completed and reserved requests")
    if identity(state["incumbent"]) != identity(incumbent) or identity(
        state["context"]
    ) != identity(context.to_dict()):
        raise ValueError("Current incumbent/context differs from the formal acceptance ledger")
    for key, value in costs.items():
        if not np.isclose(state["search_" + key], value, atol=1e-9, rtol=1e-12):
            raise ValueError("Search cost differs from evidence ledger")


def _belief(context, config, receipts, seed, bootstrap=None):
    from .belief import ResponseBelief

    belief = ResponseBelief(context, config)
    initial = seed["incumbent"]
    if seed.get("initialization", {}).get("matches_search_protocol", False):
        belief.observe_validity({**initial, "kind": "optimize"})
    shared = config.get("response_model") == "shared_projectors_v1"
    input_context = Context.from_dict(seed["context"])
    input_response = "input_response_model" in config
    if input_response and bootstrap is not None:
        if (
            bootstrap.get("context_id") != input_context.context_id
            or not bootstrap.get("valid")
            or bootstrap.get("schema") != "pas_bootstrap_input_evidence_v1"
        ):
            raise ValueError("Bootstrap response input differs from the original cold context")
        if shared or input_context.context_id == context.context_id:
            belief.bind_input_evidence(input_context, bootstrap["proposal_quadratic"])
    incumbent_quality = initial["metrics"]["quality"]
    for receipt in receipts:
        if (shared or input_response) and receipt["context_id"] != input_context.context_id:
            raise ValueError("Response history is not bound to its original input context")
        if receipt["kind"] == "survey":
            if input_response and receipt.get("valid"):
                _survey_posterior(receipt, input_context, config)
                if shared or input_context.context_id == context.context_id:
                    belief.bind_input_evidence(input_context, receipt["proposal_quadratic"])
            belief.observe_validity(receipt)
            continue
        if shared:
            belief.observe(receipt, input_context=input_context)
            # Reconstruct only the formally accepted path. A rejected output is
            # never substituted for the input of a later response observation.
            if (
                receipt["kind"] == "optimize"
                and receipt.get("valid")
                and receipt["metrics"]["quality"]
                > incumbent_quality + config["acceptance_tolerance"]
            ):
                input_context = _optimized_output(input_context, receipt)
                incumbent_quality = receipt["metrics"]["quality"]
        elif receipt["context_id"] == context.context_id:
            belief.observe(receipt)
        else:
            belief.observe_validity(receipt)
        if (
            input_response
            and not shared
            and (
                receipt["kind"] == "optimize"
                and receipt.get("valid")
                and receipt["metrics"]["quality"]
                > incumbent_quality + config["acceptance_tolerance"]
            )
        ):
            input_context = _optimized_output(input_context, receipt)
            incumbent_quality = receipt["metrics"]["quality"]
    if (shared or input_response) and input_context.context_id != context.context_id:
        raise ValueError("Current context differs from the accepted response history")
    return belief


def _audit(directory, state, manifest, callback):
    binding = manifest.get("references")
    if not binding and callback is None:
        return False
    if callback is None:
        from .acceptance import audit

        result = audit(state["incumbent"], binding["path"])
    else:
        result = callback(state["incumbent"])
    if not isinstance(result, dict) or type(result.get("passed")) is not bool:
        raise ValueError("External audit must return an explicit passed boolean")
    state["acceptance"] = result
    write_json(directory / "acceptance.json", result)
    if result["passed"]:
        state["status"] = "passed"
        state["termination"] = (
            "initialization_passed" if not state["events"] else "reference_passed"
        )
        return True
    return False


def _commit(
    directory,
    state,
    receipt,
    config,
    bootstrap=None,
    *,
    response_policy=None,
    fixed_dmrg_policy=None,
):
    planning_config = planning_configuration(config, response_policy)
    pending = state["pending"]
    context = Context.from_dict(state["context"])
    if receipt.get("context_id") != context.context_id or receipt.get("kind") != pending["kind"]:
        raise ValueError("Receipt does not belong to the dispatched input context/action")
    if tuple(sorted(receipt.get("active", []))) != tuple(pending["active"]):
        raise ValueError("Receipt active set differs from the request")
    if not receipt.get("evidence_id") or not receipt.get("request_id"):
        raise ValueError("Evidence and request identities are mandatory")
    _verify_receipt(receipt)
    if (
        pending.get("fixed_dmrg_policy") != fixed_dmrg_policy
        or fixed_dmrg_policy is None
        and "fixed_dmrg_policy" in pending
    ):
        raise ValueError("Commit fixed DMRG policy differs from its reservation")
    _verify_request(
        directory / pending["directory"],
        context,
        receipt,
        config,
        fixed_dmrg_policy=fixed_dmrg_policy,
    )
    input_posterior = _survey_posterior(receipt, context, config)
    if receipt["evidence_id"] in {e["evidence_id"] for e in state["events"]}:
        raise ValueError("The dispatched action returned already-consumed evidence")
    output = _optimized_output(context, receipt)
    costs = {}
    for field in ("cpu_seconds", "wall_seconds"):
        if field not in receipt:
            raise ValueError("Physical cost is missing; unknown is not zero")
        value = float(receipt[field])
        if not np.isfinite(value) or value < 0:
            raise ValueError("Invalid physical cost")
        costs[field] = value
    if pending["kind"] != "survey":
        _validate_metrics(receipt, config)
    accepted = False
    if pending["kind"] == "optimize" and receipt.get("valid"):
        if (
            receipt["metrics"]["quality"]
            > state["incumbent"]["metrics"]["quality"] + config["acceptance_tolerance"]
        ):
            accepted = True
    shared = config.get("response_model") == "shared_projectors_v1"
    if accepted and shared and (set(config["channels"]) & {"occupation", "composition"}):
        from .projector_response import ProjectorFrame

        ProjectorFrame(context).assert_compatible(
            ProjectorFrame(output), composition="composition" in config["channels"]
        )
    update = dict(
        context_id=context.context_id,
        evidence_id=receipt["evidence_id"],
        accepted=accepted,
        next_context_id=output.context_id if accepted else context.context_id,
    )
    if pending["kind"] == "survey":
        belief = _belief(
            context,
            planning_config,
            _read_events(directory, state),
            read_json(directory / "seed.json"),
            bootstrap,
        )
        belief.observe_validity(receipt)
        update.update(
            role="proposal_only",
            target_quality_updated=False,
            validity_updated=True,
            validity_beta=belief.validity,
            validity_scope="survey_only",
        )
        if _requires_context_survey(config):
            update["input_evidence_updated"] = input_posterior is not None
            if input_posterior is not None:
                update["input_evidence_posterior"] = {
                    key: input_posterior[key]
                    for key in (
                        "schema",
                        "context_id",
                        "basis_id",
                        "gain",
                        "raw_evidence_hash",
                        "model_hash",
                    )
                }
                if "input_response_model" in planning_config:
                    belief.bind_input_evidence(context, receipt["proposal_quadratic"])
                    update["role"] = "proposal_and_response_input"
                    update["input_response"] = belief.input_response_audit([list(context.active)])
    else:
        belief = _belief(
            context,
            planning_config,
            _read_events(directory, state),
            read_json(directory / "seed.json"),
            bootstrap,
        )
        previous_count = sum(len(v) for v in belief.observations.values())
        belief.observe(receipt)
        predictions = {}
        for channel in config["channels"]:
            mean, covariance = belief.predict_mixed(
                [receipt["active"], receipt["active"]], ["probe", "optimize"], channel
            )
            predictions[channel] = dict(
                kinds=["probe", "optimize"], mean=mean.tolist(), covariance=covariance.tolist()
            )
        update.update(
            role="conditioned_input_context",
            target_quality_updated=sum(len(v) for v in belief.observations.values())
            > previous_count,
            validity_updated=True,
            validity_beta=belief.validity,
            predictions_at_observed_candidate=predictions,
            jitter=belief.jitter_used,
            observations={k: len(v) for k, v in belief.observations.items()},
            accepted_output_starts_fresh_quality_prior=accepted
            and (not shared or not (set(config["channels"]) & {"occupation", "composition"})),
        )
        if shared:
            update.update(
                response_model="shared_projectors_v1",
                cross_context_channels=[k for k in config["channels"] if k != "qit"],
                accepted_output_resets_channels=["qit"]
                if accepted and "qit" in config["channels"]
                else [],
                quality_observation_sources={
                    k: [
                        dict(
                            context_id=r["context_id"],
                            evidence_id=r["evidence_id"],
                            kind=r["kind"],
                            active=list(r["active"]),
                        )
                        for r in rows
                    ]
                    for k, rows in belief.observations.items()
                },
            )
        if "density_response_model" in config:
            update["density_response"] = belief.density_response_audit([receipt["active"]])
        if "input_response_model" in planning_config:
            update["input_response"] = belief.input_response_audit([receipt["active"]])
    write_json(directory / "filter_updates" / f"{pending['number']:03d}.json", update)
    relative = str(Path(pending["directory"]) / "receipt.json")
    write_json(directory / relative, receipt)
    if accepted:
        state["incumbent"] = receipt
        state["context"] = output.to_dict()
        if output.context_id != context.context_id:
            state["contexts"].append(output.context_id)
    if pending["kind"] == "survey":
        state["survey_collected"] = True
    state["events"].append(
        dict(
            number=pending["number"],
            kind=pending["kind"],
            context_id=context.context_id,
            evidence_id=receipt["evidence_id"],
            receipt=relative,
            sha256=file_hash(directory / relative),
            accepted=accepted,
            valid=bool(receipt.get("valid")),
        )
    )
    for field, value in costs.items():
        state["search_" + field] += value
    state["pending"] = None
    state["status"] = "running"
    write_json(directory / "state.json", state)


def _summary(state):
    return dict(
        status=state["status"],
        termination=state.get("termination"),
        quality=state["incumbent"]["metrics"]["quality"],
        metrics=state["incumbent"]["metrics"],
        remaining_diagnostics=state["remaining_diagnostics"],
        remaining_optimizations=state["remaining_optimizations"],
        executions=len(state["events"]),
        accepted=sum(e["accepted"] for e in state["events"]),
        initialization=state["initialization"],
        search_cpu_seconds=state["search_cpu_seconds"],
        search_wall_seconds=state["search_wall_seconds"],
        planning_wall_seconds=state["planning_wall_seconds"],
        acceptance=state.get("acceptance"),
        pending=state["pending"],
    )


def run(directory, *, execute=None, choose=None, audit_callback=None, max_actions=None):
    directory = Path(directory).resolve()
    import fcntl

    with (directory / "worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = _verify_manifest(directory)
        config = configuration(manifest["config"])
        response_policy = manifest.get("response_policy")
        window_policy = manifest.get("survey_window_policy")
        planning_config = planning_configuration(config, response_policy)
        state = read_json(directory / "state.json")
        seed = read_json(directory / "seed.json")
        _verify_receipt(seed["incumbent"])
        _verify_receipt(state["incumbent"])
        _verify_state(directory, state, manifest, seed)
        bootstrap = _bootstrap_input(directory, manifest)
        if state["status"] in ("passed", "budget_exhausted", "exhausted"):
            if state["status"] == "passed" and manifest.get("references"):
                if not _audit(directory, state, manifest, audit_callback):
                    raise ValueError(
                        "Previously passed incumbent no longer passes the bound reference audit"
                    )
            return state
        if execute is None:
            execute = _backend(config).execute
        default_planner = choose is None
        if default_planner:
            from .planner import choose
        write_json(
            directory / "worker.json",
            dict(pid=os.getpid(), started_unix=time.time(), manifest_id=manifest["manifest_id"]),
        )
        dispatched = 0
        try:
            while True:
                if state["pending"]:
                    fixed_policy = _fixed_dmrg_policy(manifest, state["pending"]["number"])
                    physical_options = (
                        {} if fixed_policy is None else dict(fixed_dmrg_policy=fixed_policy)
                    )
                    path = directory / state["pending"]["directory"] / "receipt.json"
                    if not path.exists():
                        backend = _backend(config)
                        if not hasattr(backend, "reconcile"):
                            raise RuntimeError(
                                "Unresolved request; backend has no read-only reconciliation; automatic rerun prohibited"
                            )
                        pending = state["pending"]
                        receipt = backend.reconcile(
                            Context.from_dict(state["context"]),
                            pending["active"],
                            pending["kind"],
                            path.parent,
                            config,
                            **physical_options,
                        )
                        write_json(path, receipt)
                    _commit(
                        directory,
                        state,
                        read_json(path),
                        config,
                        bootstrap,
                        response_policy=response_policy,
                        fixed_dmrg_policy=fixed_policy,
                    )
                if _audit(directory, state, manifest, audit_callback):
                    break
                if (directory / "hold.json").exists():
                    hold = read_json(directory / "hold.json")
                    if hold["manifest_id"] != manifest["manifest_id"]:
                        raise ValueError("Hold belongs to another run")
                    state.update(status="held", termination="held_before_next_dispatch", hold=hold)
                    break
                if state["remaining_optimizations"] <= 0:
                    state.update(
                        status="budget_exhausted", termination="optimization_budget_exhausted"
                    )
                    break
                if max_actions is not None and dispatched >= max_actions:
                    state["status"] = "ready"
                    break
                context = Context.from_dict(state["context"])
                receipts = _read_events(directory, state)
                fixed_policy = _fixed_dmrg_policy(manifest, len(state["events"]) + 1)
                belief = _belief(context, planning_config, receipts, seed, bootstrap)
                excluded = {
                    kind: [
                        r["active"]
                        for r in receipts
                        if r["kind"] == kind and r["context_id"] == context.context_id
                    ]
                    for kind in ("probe", "optimize")
                }
                domain, survey, bootstrap_active = _input_domain(
                    context, bootstrap, receipts, fixed_dmrg_policy=fixed_policy
                )
                start = time.perf_counter()
                if _requires_context_survey(config) and survey is None:
                    attempted = any(
                        r["kind"] == "survey"
                        and r["context_id"] == context.context_id
                        and r.get("fixed_dmrg_policy") == fixed_policy
                        for r in receipts
                    )
                    if attempted:
                        decision = dict(kind="stop", reason="context_survey_failed_no_retry")
                    elif state["remaining_diagnostics"] <= 0:
                        decision = dict(
                            kind="stop", reason="context_survey_diagnostic_budget_exhausted"
                        )
                    else:
                        window = (
                            derive_window(context, window_policy)
                            if window_policy is not None
                            else None
                        )
                        decision = dict(
                            kind="survey",
                            active=window["active"] if window is not None else list(context.window),
                            audit=dict(
                                stage="current_context_evidence_collection",
                                context_id=context.context_id,
                            ),
                        )
                        if window is not None:
                            decision["audit"]["survey_window"] = window
                elif (
                    not _requires_context_survey(config)
                    and config["full_window_survey"]
                    and not state["survey_collected"]
                    and state["remaining_diagnostics"] > 0
                ):
                    decision = dict(
                        kind="survey",
                        active=list(context.window),
                        audit={"stage": "initial_evidence_collection"},
                    )
                else:
                    # Policy is a manifest-level planning binding, independent of
                    # immutable solver configurations and physical receipts.
                    policy_arguments = (
                        dict(planner_policy=manifest.get("planner_policy"))
                        if default_planner
                        else {}
                    )
                    decision = choose(
                        domain,
                        belief,
                        state["incumbent"]["metrics"]["quality"],
                        state["remaining_diagnostics"],
                        state["remaining_optimizations"],
                        planning_config,
                        event_index=len(state["events"]),
                        excluded=excluded,
                        survey=survey,
                        **policy_arguments,
                    )
                if bootstrap_active:
                    decision.setdefault("audit", {})["bootstrap_input_evidence"] = dict(
                        source=bootstrap["source"],
                        measured_window=bootstrap["measured_window"],
                        physical_window=list(context.window),
                        current_protocol_receipt=False,
                        response_observation=False,
                        validity_observation=False,
                        diagnostic_budget_charged=False,
                        event_index=len(state["events"]),
                    )
                state["planning_wall_seconds"] += time.perf_counter() - start
                number = len(state["events"]) + 1
                if fixed_policy is not None:
                    decision.setdefault("audit", {})["fixed_dmrg_policy"] = deepcopy(fixed_policy)
                write_json(directory / "decisions" / f"{number:03d}.json", decision)
                if decision["kind"] == "stop":
                    state.update(
                        status="exhausted",
                        termination=decision.get("reason", "candidate_requests_exhausted"),
                    )
                    break
                kind = decision["kind"]
                active = sorted(decision["active"])
                if kind not in ("probe", "optimize", "survey"):
                    raise ValueError("Unknown planner action")
                if _requires_context_survey(config):
                    attempts = {
                        _survey_key(r["context_id"], r.get("fixed_dmrg_policy")): bool(
                            r.get("valid")
                        )
                        for r in receipts
                        if r["kind"] == "survey"
                    }
                    _verify_survey_action(
                        kind,
                        context.context_id,
                        attempts,
                        bootstrap=bootstrap_active,
                        fixed_dmrg_policy=fixed_policy,
                    )
                if kind != "survey":
                    domain.partition(active)
                    if active == sorted(context.active):
                        raise ValueError(
                            "Current retained space cannot be dispatched as a new candidate"
                        )
                    if active in [sorted(a) for a in excluded[kind]]:
                        raise ValueError(
                            "Already attempted request cannot be dispatched again in this context"
                        )
                budget = (
                    "remaining_optimizations" if kind == "optimize" else "remaining_diagnostics"
                )
                if state[budget] <= 0:
                    raise ValueError("Planner attempted to overspend budget")
                # Reserve durably before dispatch; interrupted reservations are reconciled,
                # not refunded or retried without proof about physical execution.
                state[budget] -= 1
                state["pending"] = dict(
                    number=number,
                    kind=kind,
                    active=active,
                    directory=f"actions/{number:03d}_{kind}",
                    context_id=context.context_id,
                    reserved_unix=time.time(),
                )
                if fixed_policy is not None:
                    state["pending"]["fixed_dmrg_policy"] = deepcopy(fixed_policy)
                state["status"] = "running"
                write_json(directory / "state.json", state)
                action_directory = directory / state["pending"]["directory"]
                action_directory.mkdir(parents=True, exist_ok=True)
                physical_options = (
                    {} if fixed_policy is None else dict(fixed_dmrg_policy=fixed_policy)
                )
                receipt = execute(
                    context, active, kind, action_directory, config, **physical_options
                )
                write_json(action_directory / "receipt.json", receipt)
                _commit(
                    directory,
                    state,
                    receipt,
                    config,
                    bootstrap,
                    response_policy=response_policy,
                    fixed_dmrg_policy=fixed_policy,
                )
                dispatched += 1
        except BaseException as exc:
            state.update(
                status="halted",
                failure=dict(
                    type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    automatic_retry=False,
                    time_unix=time.time(),
                ),
            )
            write_json(directory / "state.json", state)
            write_json(directory / "summary.json", _summary(state))
            raise
        finally:
            write_json(
                directory / "worker_finished.json", dict(pid=os.getpid(), time_unix=time.time())
            )
        state["updated_unix"] = time.time()
        write_json(directory / "state.json", state)
        write_json(directory / "summary.json", _summary(state))
        return state
