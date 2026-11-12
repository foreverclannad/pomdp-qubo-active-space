"""Offline baseline planner replay; no optional feedback policy is enabled."""

from pas import planner
from pas.common import file_hash, identity, read_json
from pas.config import planning_configuration
from pas.contracts import Context

from .. import engine
from ..common import Evidence, require


def _compare(actual, expected):
    for key in ("kind", "active", "pool"):
        if actual.get(key) != expected.get(key):
            raise AssertionError("Baseline does not reproduce recorded " + key)
    error = 0.0
    left, right = (d["audit"]["optimize_values"] for d in (actual, expected))
    for a, b in zip(left, right, strict=True):
        if a["active"] != b["active"]:
            raise AssertionError("H2 candidate order changed")
        for key in (
            "value",
            "immediate_valid_expected_gain",
            "valid_rejection_continuation",
            "invalid_continuation",
        ):
            error = max(error, abs(a[key] - b[key]))
    if error > 1e-12:
        raise AssertionError("Baseline H2 values differ from recorded values: " + str(error))
    return error


def replay(evidence_directory, original_run):
    evidence = Evidence(evidence_directory)
    run = evidence.tree / original_run.lstrip("/")
    before = {str(p): file_hash(p) for p in sorted(run.rglob("*.json"))}
    manifest, seed, state = [
        read_json(run / (name + ".json")) for name in ("manifest", "seed", "state")
    ]
    if (
        identity({k: v for k, v in manifest.items() if k != "manifest_id"})
        != manifest["manifest_id"]
        or identity(seed) != manifest["seed_id"]
        or state["manifest_id"] != manifest["manifest_id"]
    ):
        raise ValueError("Run manifest/seed/state binding failed")
    cfg = planning_configuration(manifest["config"], manifest.get("response_policy"))
    if "input_response_model" not in cfg or "input_feedback_model" in cfg:
        raise ValueError("Replay requires a baseline input-response run without feedback options")
    bootstrap = (
        evidence.read(
            manifest["bootstrap_input_evidence"]["path"],
            manifest["bootstrap_input_evidence"]["sha256"],
        )
        if "bootstrap_input_evidence" in manifest
        else None
    )
    context = evidence.context(seed["context"])
    evidence.receipt(seed["incumbent"])
    seed = evidence.relocate(seed)
    bootstrap = evidence.relocate(bootstrap)
    receipts, rows = ([], [])
    quality = seed["incumbent"]["metrics"]["quality"]
    budgets = engine.budgets(manifest)
    nd, no = (budgets["diagnostic_budget"], budgets["optimization_budget"])
    with engine.read_only():
        for event in state["events"]:
            n = event["number"]
            path = run / event["receipt"]
            if n != len(receipts) + 1 or file_hash(path) != event["sha256"]:
                raise ValueError("Event order or receipt hash changed")
            r = read_json(path)
            if r["context_id"] != context.context_id or r["evidence_id"] != event["evidence_id"]:
                raise ValueError("Receipt belongs to another context or evidence identity")
            evidence.receipt(r)
            if r["kind"] == "optimize":
                actual = read_json(run / "decisions" / f"{n:03d}.json")
                if actual["active"] != r["active"] or actual["kind"] != r["kind"]:
                    raise ValueError("Recorded dispatch differs from the physical receipt")
                base = engine.belief(context, cfg, receipts, seed, bootstrap)
                fixed = engine.fixed_policy(manifest, n)
                domain, survey, _ = engine.input_domain(
                    context, bootstrap, receipts, fixed_dmrg_policy=fixed
                )
                excluded = {
                    kind: [
                        a["active"]
                        for a in receipts
                        if a["kind"] == kind and a["context_id"] == context.context_id
                    ]
                    for kind in ("probe", "optimize")
                }
                args = dict(
                    event_index=len(receipts),
                    excluded=excluded,
                    survey=survey,
                    planner_policy=manifest["planner_policy"],
                )
                old = planner.choose(domain, base, quality, nd, no, cfg, **args)
                baseline_error = _compare(old, actual)
                row = dict(
                    event=n,
                    context_id=context.context_id,
                    original_active=r["active"],
                    original_accepted=event["accepted"],
                    original_metrics=r["metrics"],
                    baseline_max_h2_error=baseline_error,
                    same_dispatch=True,
                    same_pool=True,
                )
                rows.append(row)
            accepted = bool(
                r["kind"] == "optimize"
                and r["valid"]
                and (r["metrics"]["quality"] > quality + cfg["acceptance_tolerance"])
            )
            if accepted != event["accepted"]:
                raise AssertionError("Recorded acceptance rule differs")
            if accepted:
                context = evidence.optimized(context, r)
                quality = r["metrics"]["quality"]
            receipts.append(evidence.relocate(r))
            if r["kind"] == "optimize":
                no -= 1
            else:
                nd -= 1
    after = {str(p): file_hash(p) for p in sorted(run.rglob("*.json"))}
    if before != after:
        raise RuntimeError("Source run changed during the read-only replay")
    evidence.unchanged()
    require(
        context.context_id == Context.from_dict(state["context"]).context_id,
        "Planner replay ended in a different orbital context",
    )
    require(
        (nd, no) == (state["remaining_diagnostics"], state["remaining_optimizations"]),
        "Planner replay ended with different remaining budgets",
    )
    require(state["pending"] is None, "Archived run has an unresolved physical request")
    return dict(
        schema="pas_paper_baseline_replay_v1",
        run=original_run,
        manifest_id=manifest["manifest_id"],
        original_source_id=manifest["source_id"],
        effective_run_budgets=budgets,
        events_replayed=len(receipts),
        rounds=rows,
        original_final_metrics=state["incumbent"]["metrics"],
        physical_jobs_launched=0,
        source_run_unchanged=True,
        passed=True,
    )
