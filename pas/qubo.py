"""QUBO proposals in the hard orbital/reference-electron feasible domain."""

from __future__ import annotations

import ctypes
import hashlib
import heapq
import itertools
import math
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from .belief import Quadratic
from .common import derived_seed, identity


def seed_context(context, config):
    """Disabled source values cannot affect decisions indirectly through RNG keys."""
    fields = context.to_dict()
    fields.pop("payload")
    if "composition" not in config["channels"]:
        fields.pop("composition")
        fields.pop("composition_pair", None)
        fields.pop("composition_channels", None)
        fields.pop("composition_preferences", None)
    return identity(fields)


def _groups(context):
    required = set(context.required_active)
    if not required <= set(context.window):
        raise ValueError("Proposal window must contain every required active orbital")
    window = tuple(i for i in context.window if i not in required)
    k = context.k - len(required)
    if not context.electron_roles:
        return ((window, k),)
    groups = [tuple(i for i in window if context.electron_roles[i] == role) for role in range(3)]
    singles = len(groups[1])
    electrons = context.electrons - sum(context.electron_roles[i] for i in required)
    doubles = (electrons - singles) // 2
    return tuple(zip(groups, (k - singles - doubles, singles, doubles)))


def candidate_count(context):
    """Count the same legal domain used by enumeration and native annealing."""
    return math.prod(math.comb(len(group), count) for group, count in _groups(context))


def feasible_candidates(context):
    for pieces in itertools.product(
        *(itertools.combinations(group, count) for group, count in _groups(context))
    ):
        yield tuple(
            sorted(itertools.chain(context.required_active, itertools.chain.from_iterable(pieces)))
        )


def _legal(context, active):
    try:
        context.partition(active)
        return True
    except (ValueError, TypeError):
        return False


@lru_cache(maxsize=1)
def _native():
    source = Path(__file__).with_name("anneal.c")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pomdp-active-space"
    cache.mkdir(parents=True, exist_ok=True)
    library = cache / f"anneal-{digest}.so"
    if not library.exists():
        handle, temporary = tempfile.mkstemp(prefix="anneal-", suffix=".so", dir=cache)
        os.close(handle)
        try:
            subprocess.run(
                [
                    os.environ.get("CC", "cc"),
                    "-O3",
                    "-std=c99",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-lm",
                    "-o",
                    temporary,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            os.replace(temporary, library)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(
                "The million-step proposal solver requires a working C compiler"
            ) from exc
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    function = ctypes.CDLL(str(library)).pas_anneal
    double_pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    int_pointer = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        double_pointer,
        double_pointer,
        int_pointer,
        ctypes.c_int,
        int_pointer,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_int,
        int_pointer,
        double_pointer,
        ctypes.POINTER(ctypes.c_uint64),
    ]
    function.restype = ctypes.c_int
    return function


def solve(context, objective: Quadratic, config, seed, excluded=(), collect=1):
    """One objective, one fixed SA budget, and up to collect visited alternatives."""
    required = tuple(sorted(context.required_active))
    window = tuple(i for i in context.window if i not in set(required))
    k = context.k - len(required)
    forbidden = {tuple(sorted(a)) for a in excluded if _legal(context, a)}
    if type(collect) is not int or collect < 1:
        raise ValueError("Visited-candidate capacity must be a positive integer")
    if (
        np.shape(objective.linear) != (context.nmo,)
        or np.shape(objective.pair) != (context.nmo, context.nmo)
        or not np.isfinite(objective.linear).all()
        or not np.isfinite(objective.pair).all()
        or not np.isfinite(objective.constant)
    ):
        raise ValueError("A finite QUBO over the complete orbital basis is required")
    count = candidate_count(context)
    if count <= config["exact_limit"] or k == 0:
        retained, examined = [], 0
        for active in feasible_candidates(context):
            examined += 1
            if active in forbidden:
                continue
            value = objective(active)
            # Negative indices make equal-score selection lexicographically stable.
            entry = (value, tuple(-i for i in active), active)
            if len(retained) < collect:
                heapq.heappush(retained, entry)
            elif entry > retained[0]:
                heapq.heapreplace(retained, entry)
        visited = [row[2] for row in sorted(retained, reverse=True)]
        winner = visited[0] if visited else None
        return winner, dict(
            solver="enumeration",
            combinations=count,
            examined=examined,
            steps=0,
            score=objective(winner) if winner is not None else None,
            visited_candidates=[list(a) for a in visited],
            global_optimum_proven=True,
            exhausted=winner is None,
        )
    if len(forbidden) == count:
        return None, dict(
            solver="request_exhaustion",
            combinations=count,
            steps=0,
            score=None,
            visited_candidates=[],
            global_optimum_proven=False,
            exhausted=True,
        )
    local = {orbital: index for index, orbital in enumerate(window)}
    excludes = np.asarray(
        [[local[i] for i in a if i not in required] for a in sorted(forbidden)], dtype=np.int32
    ).reshape(-1, k)
    pair = np.asarray(objective.pair[np.ix_(window, window)], dtype=np.float64)
    pair = np.ascontiguousarray((pair + pair.T) / 2)
    linear = np.ascontiguousarray(objective.linear[list(window)] + np.diag(pair), dtype=np.float64)
    if required:
        # Fix x_core=1 exactly, preserving both directions of the full xQx terms.
        linear += objective.pair[np.ix_(window, required)].sum(axis=1) + objective.pair[
            np.ix_(required, window)
        ].sum(axis=0)
    np.fill_diagonal(pair, 0)
    temperature = max(float(np.std(linear) + 2 * math.sqrt(k) * np.std(pair)), 1e-6)
    output = np.zeros((collect, k), dtype=np.int32)
    scores, completed = np.zeros(collect, dtype=np.float64), ctypes.c_uint64()
    roles = np.ascontiguousarray(
        [context.electron_roles[i] for i in window]
        if context.electron_roles
        else np.zeros(len(window)),
        dtype=np.int32,
    )
    electrons = (
        context.electrons - sum(context.electron_roles[i] for i in required)
        if context.electron_roles
        else -1
    )
    found = _native()(
        len(window),
        k,
        config["annealing_steps"],
        config["annealing_restarts"],
        int(seed) % 2**64,
        linear,
        pair,
        roles,
        electrons,
        excludes,
        len(excludes),
        temperature,
        collect,
        output,
        scores,
        ctypes.byref(completed),
    )
    if found < 0:
        raise RuntimeError(f"Native annealing failed: {found}")
    if completed.value != config["annealing_steps"]:
        raise RuntimeError("Native solver did not execute the frozen per-objective step budget")
    visited = [
        tuple(sorted(required + tuple(window[int(i)] for i in row))) for row in output[:found]
    ]
    visited.sort(key=lambda a: (-objective(a), a))
    for active in visited:
        context.partition(active)
    active = visited[0] if visited else None
    return active, dict(
        solver="annealing",
        combinations=count,
        steps=completed.value,
        restarts=config["annealing_restarts"],
        steps_per_restart=completed.value // config["annealing_restarts"],
        visited_candidates=[list(a) for a in visited],
        global_optimum_proven=False,
        score=objective(active) if found else None,
        exhausted=False,
    )


def _posterior_capture_pool(context, config, event_index, excluded, survey, belief=None):
    """One posterior-mean objective, retaining distinct visited alternatives."""
    if survey is None or survey.get("context_id") != context.context_id or not survey.get("valid"):
        raise ValueError("Posterior capture needs a valid survey in this orbital context")
    fields = survey.get("proposal_quadratic", {})
    from .evidence_belief import EvidenceResponse, validate_proposal_score_model

    validate_proposal_score_model(fields, config)
    posterior = fields.get("input_posterior", {})
    if (
        fields.get("score_model") != "posterior_evidence_capture_v1"
        or fields.get("composition_scale") != "terminal_score"
        or posterior.get("schema") != "pas_capture_statistics_posterior_v1"
        or posterior.get("context_id") != context.context_id
        or posterior.get("basis_id") != context.basis_id
        or fields.get("quadratic_convention") != "xQx"
        or not set(context.window) <= set(fields.get("measured_indices", []))
    ):
        raise ValueError(
            "Posterior capture requires filtered, complete measured-window coefficients"
        )
    feedback = None
    if config.get("input_feedback_proposals"):
        if belief is None or belief.context.context_id != context.context_id:
            raise ValueError("Feedback proposals require the bound current-context response belief")
        feedback = belief.input_posterior()
        fields = feedback.proposal_fields(fields, config)
    objective = Quadratic(
        float(fields["constant"]),
        np.asarray(fields["linear"], float),
        np.asarray(fields["quadratic"], float),
    )
    seed = derived_seed(config["seed"], seed_context(context, config), event_index, "capture_pool")
    capacity = 2 * config["pool_size"] - 1
    # Reuse the former total search budget; every restart now optimizes the same
    # input posterior, with no output-response objectives or first-step override.
    search = dict(
        config,
        annealing_steps=config["annealing_steps"] * config["pool_size"],
        annealing_restarts=config["annealing_restarts"] * config["pool_size"],
    )
    _, record = solve(
        context, objective, search, seed, (excluded or {}).get("optimize", ()), collect=capacity
    )
    pool = [tuple(a) for a in record["visited_candidates"]]
    if not pool and not record["exhausted"]:
        raise RuntimeError(
            "Fixed-budget posterior capture found no eligible optimization; no automatic retry"
        )
    audit = dict(
        seed=seed,
        objectives=[dict(source="input_posterior_capture_mean", **record)],
        requested_size=capacity,
        actual_size=len(pool),
        maximum_pool_size=capacity,
        excluded_current_active=list(sorted(context.active)),
        total_annealing_steps=record["steps"],
        objective_count=1,
        survey_is_target_observation=False,
        input_posterior=posterior,
        output_response_used_for_proposals=False,
        first_action_override=False,
    )
    if feedback is not None:
        audit.update(
            output_response_used_for_proposals=True,
            joint_input_posterior=fields["joint_input_posterior"],
            proposal_source="joint_conditioned_input_mean_not_total_response_mean",
        )
    if config.get("proposal_score_model") == "terminal_aligned_v1":
        response = feedback if feedback is not None else EvidenceResponse(context, fields)
        scores = response.proposal_scores(pool, config["weights"])
        rows = [
            dict(
                active=list(a),
                surrogate_score=objective(a),
                exact_score=float(scores["score"][i]),
                terminal_score=float(scores["terminal_score"][i]),
                cross_boundary_qit=float(scores["cross_boundary_qit"][i]),
                channels={c: float(values[i]) for c, values in scores["channels"].items()},
                surrogate_gap=float(objective(a) - scores["score"][i]),
            )
            for i, a in enumerate(pool)
        ]
        if any(
            not np.isfinite(row["exact_score"]) or row["surrogate_gap"] < -1e-10 for row in rows
        ):
            raise ValueError(
                "Proposal surrogate is not a finite supporting bound of the exact score"
            )
        rows.sort(key=lambda row: (-row["exact_score"], row["active"]))
        pool = [tuple(row["active"]) for row in rows]
        audit["candidate_rescoring"] = dict(
            model="terminal_aligned_v1",
            scope="retained_QUBO_pool_only",
            target_quality_observation=False,
            global_exact_optimum_proven=False,
            extra_annealing_steps=0,
            rows=rows,
        )
        audit["objectives"][0]["score_scope"] = "supporting_quadratic_not_exact_nonlinear_score"
    return pool, audit


def propose(context, belief, config, event_index=0, excluded=None, survey=None):
    """Fixed objective slots; duplicate winners use already evaluated candidates.

    The retained space is a baseline, never a new candidate. A previous H request
    on another space may still be a useful L request, so the legacy pool removes
    other candidates only when both requests are consumed. At least one H
    request remains available in every nonempty pool unless that domain is exhausted.
    Append the mean objective's retained alternatives after the original pool,
    without spending additional objectives, random draws, or annealing steps.
    """
    excluded = {
        kind: {tuple(sorted(a)) for a in (excluded or {}).get(kind, []) if _legal(context, a)}
        for kind in ("probe", "optimize")
    }
    for candidates in excluded.values():
        candidates.add(tuple(sorted(context.active)))
    if config.get("survey_model") == "posterior_evidence_capture_v1":
        return _posterior_capture_pool(context, config, event_index, excluded, survey, belief)
    completed_both = excluded["probe"] & excluded["optimize"]
    seed = derived_seed(config["seed"], seed_context(context, config), event_index, "pool")
    rng = np.random.default_rng(seed)
    count = config["pool_size"]
    mean = belief.utility_quadratic()
    objectives = [("posterior_mean", mean)]
    evidence = None
    if survey is not None:
        if survey.get("context_id") != context.context_id or not survey.get("valid"):
            raise ValueError("Survey proposal requires valid evidence in this orbital frame")
        if not set(config["channels"]) & {"occupation", "qit"}:
            raise ValueError("Known composition alone does not justify a survey")
        fields = survey.get("proposal_quadratic")
        if fields is not None:
            if fields.get("candidate_scope") == "must_be_subset_of_measured_indices" and not set(
                context.window
            ) <= set(fields.get("measured_indices", [])):
                raise ValueError("Proposal domain contains unmeasured survey orbitals")
            linear = np.asarray(fields["linear"], dtype=float)
            pair = np.asarray(fields["quadratic"], dtype=float)
            if linear.shape != (context.nmo,) or pair.shape != (context.nmo, context.nmo):
                raise ValueError("Survey QUBO must cover the full MO basis")
            if (
                fields.get("quadratic_convention", survey.get("quadratic_convention", "xQx"))
                != "xQx"
            ):
                raise ValueError("Survey QUBO must use the full-matrix xQx convention")
            if count > 1:
                objectives.append(
                    ("survey_proposal", Quadratic(float(fields.get("constant", 0)), linear, pair))
                )
        elif survey.get("proposal_active") is not None:
            evidence = tuple(context.partition(survey["proposal_active"])["active"])
        else:
            raise ValueError("A survey must supply a bound evidence proposal")
    # An evidence slot replaces exactly one posterior draw even when its winner
    # duplicates the mean. It never causes a ninth annealing objective.
    literal_slot = evidence is not None and count > 1
    while len(objectives) + int(literal_slot) < count:
        objectives.append(("posterior_sample", belief.utility_quadratic(rng)))
    pool, records, alternatives = [], [], []
    for index, (source, objective) in enumerate(objectives):
        active, audit = solve(
            context,
            objective,
            config,
            derived_seed(seed, index),
            completed_both,
            collect=count + len(excluded["optimize"]),
        )
        visited = [tuple(a) for a in audit["visited_candidates"]]
        records.append(dict(source=source, **audit))
        alternatives.extend(visited)
        if active is not None and active not in pool:
            pool.append(active)
    if literal_slot:
        evidence = None if evidence in completed_both else evidence
        records.insert(
            1,
            dict(
                source="survey_proposal",
                solver="physical_evidence_proposal",
                steps=0,
                active=list(evidence) if evidence else None,
                global_optimum_proven=False,
            ),
        )
        if evidence is not None:
            alternatives.insert(0, evidence)
            if evidence not in pool:
                pool.insert(1 if pool else 0, evidence)
    for active in alternatives:
        if len(pool) >= count:
            break
        if active not in pool:
            pool.append(active)
    # Keep useful diagnostic-only candidates, but never stop merely because all
    # QUBO winners have already consumed H. Reuse a visited H-eligible candidate.
    h_fallback = None
    if not any(a not in excluded["optimize"] for a in pool):
        replacement = next((a for a in alternatives if a not in excluded["optimize"]), None)
        if replacement is None:
            if len(excluded["optimize"]) < candidate_count(context):
                # A finite stochastic search can fail to visit an eligible H.
                # Do not hide that failure behind an extra proposal heuristic,
                # additional annealing, or a scientific convergence claim.
                raise RuntimeError(
                    "Fixed-budget proposal search visited no unattempted optimization; "
                    "legal requests remain, so automatic stop/retry is prohibited"
                )
        else:
            h_fallback = dict(
                source="visited_candidate_coverage", active=list(replacement), steps=0
            )
        if replacement is not None:
            if len(pool) == count:
                pool[-1] = replacement
            else:
                pool.append(replacement)
    legacy_pool = list(pool)
    mean_candidates = [tuple(a) for a in records[0]["visited_candidates"][:count]]
    maximum_pool_size = 2 * count - 1
    omitted = []
    for active in mean_candidates:
        if active not in pool:
            if len(pool) < maximum_pool_size:
                pool.append(active)
            else:
                # At size one, the H-coverage fallback may replace the mean winner.
                # Keep that eligible action rather than violating the pool bound.
                omitted.append(list(active))
    audit = dict(
        seed=seed,
        objectives=records,
        requested_size=count,
        excluded_current_active=list(sorted(context.active)),
        actual_size=len(pool),
        survey_is_target_observation=False,
        final_candidate_count=len(pool),
        maximum_pool_size=maximum_pool_size,
        total_annealing_steps=sum(r["steps"] for r in records),
        objective_count=len(records),
        extra_annealing_objectives=0,
        optimization_coverage_fallback=h_fallback,
        pool_expansion=dict(
            schema="pas_mean_visited_union_v1",
            legacy_pool=[list(a) for a in legacy_pool],
            mean_candidates=[list(a) for a in mean_candidates],
            added_candidates=[list(a) for a in pool[len(legacy_pool) :]],
            omitted_for_capacity=omitted,
        ),
    )
    if "composition" in config["channels"] and config["composition_model"] in (
        "harmonic_determinant_pair_v1",
        "harmonic_channel_pair_v1",
    ):
        audit["composition_proposal"] = belief.composition_proposal_audit()
    return pool, audit
