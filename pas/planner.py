"""Frozen diagnostic/optimization and optimize-only local planning policies."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np
from scipy.special import log_ndtr, ndtr, ndtri_exp

from .belief import Quadratic
from .common import derived_seed
from .config import configuration
from .qubo import propose, seed_context


def fixed_planner_policy(policy=None):
    """Validate a separately frozen planning policy, never a solver parameter."""
    if policy is None:
        policy = dict(schema="pas_fixed_cost_ratio_v1", fixed_cost_ratio=8.0)
    if isinstance(policy, dict) and policy.get("schema") in {
        "pas_optimize_only_two_step_v1",
        "pas_survey_first_then_two_step_v1",
    }:
        if (
            set(policy) != {"schema", "depth"}
            or type(policy.get("depth")) is not int
            or policy["depth"] != 2
        ):
            raise ValueError("Optimize-only planner policy schema requires exactly depth=2")
        return dict(schema=policy["schema"], depth=2)
    if (
        not isinstance(policy, dict)
        or set(policy) != {"schema", "fixed_cost_ratio"}
        or policy.get("schema") != "pas_fixed_cost_ratio_v1"
    ):
        raise ValueError("Unknown fixed-cost planner policy schema")
    ratio = policy["fixed_cost_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, Real):
        raise ValueError("Fixed optimization/diagnostic cost ratio must be finite and positive")
    try:
        ratio = float(ratio)
    except (OverflowError, ValueError) as error:
        raise ValueError(
            "Fixed optimization/diagnostic cost ratio must be finite and positive"
        ) from error
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("Fixed optimization/diagnostic cost ratio must be finite and positive")
    return dict(schema=policy["schema"], fixed_cost_ratio=ratio)


def expected_improvement(mean, variance, incumbent, tolerance=1e-6):
    """E[(Y-incumbent) 1(Y>incumbent+tolerance)] for a Gaussian Y.

    No clipping of Gaussian observations is hidden in the conditioning model.
    Gaussian tails and the fixed quadratic response family are model assumptions.
    """
    mean, variance = np.broadcast_arrays(
        np.asarray(mean, dtype=float), np.asarray(variance, dtype=float)
    )
    if np.any(variance < -1e-9):
        raise ValueError("Negative predictive variance")
    sd = np.sqrt(np.maximum(variance, 0))
    difference = mean - float(incumbent)
    result = np.maximum(difference, 0) * (difference > tolerance)
    positive = sd > 1e-12
    if np.any(positive):
        z = (difference[positive] - tolerance) / sd[positive]
        cdf = np.asarray([0.5 * math.erfc(-float(v) / math.sqrt(2)) for v in z])
        density = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        result = np.array(result, copy=True)
        result[positive] = difference[positive] * cdf + sd[positive] * density
    return np.maximum(result, 0)


def action_values(belief, pool, incumbent_quality, config, normal_samples):
    """Return immediate-H and probe-then-H gains; rows retain source independence."""
    channels = config["channels"]
    weights = np.asarray([config["weights"][c] for c in channels], dtype=float)
    weights /= weights.sum()
    n = len(pool)
    high = {}
    mean, variance = np.zeros(n), np.zeros(n)
    for channel, weight in zip(channels, weights):
        mu, cov = belief.predict(pool, "optimize", channel)
        high[channel] = (mu, cov)
        mean += weight * mu
        variance += weight**2 * np.diag(cov)
    ph, pl = belief.success_probability("optimize"), belief.success_probability("probe")
    direct = ph * expected_improvement(
        mean, variance, incumbent_quality, config["acceptance_tolerance"]
    )
    return direct, high, weights, ph, pl


def _rejection_normals(normals, weighted_sd, boundary):
    """Gaussian channel innovations conditional on their weighted rejection.

    A deterministic orthogonal Householder construction aligns coordinate zero
    with the quality direction. Its normal quantile is truncated above the
    acceptance boundary; orthogonal coordinates retain their full distribution.
    """
    scale = float(np.linalg.norm(weighted_sd))
    if scale == 0:
        return normals, float(boundary >= 0)
    threshold = boundary / scale
    probability = float(ndtr(threshold))
    if probability == 0:
        return None, 0.0
    unit = np.asarray(weighted_sd, dtype=float) / scale
    # Q = 2 vv^T/(v^Tv)-I maps e_0 onto unit. Using e_0+unit is
    # stable even when the nonnegative quality direction almost equals e_0.
    v = unit.copy()
    v[0] += 1
    aligned = normals.copy()
    # Equivalent to ndtri(ndtr(z_0)*Phi(threshold)), evaluated in log space
    # so tiny but nonzero rejection probabilities do not produce -infinity.
    aligned[:, 0] = ndtri_exp(log_ndtr(normals[:, 0]) + log_ndtr(threshold))
    return 2 * np.outer(aligned @ v, v) / (v @ v) - aligned, probability


def _choose_optimize_only(
    belief,
    pool,
    h_indices,
    direct,
    high,
    weights,
    ph,
    incumbent,
    remaining,
    cfg,
    normals,
    policy,
    policy_source,
    generation,
    seed,
):
    """Two H actions, terminating a simulated branch at its first acceptance.

    The first action's immediate reward and invalid branch are integrated
    analytically. Only the valid-rejection continuation uses Gaussian fantasies.
    No output orbital basis is invented and no physical receipt is fabricated.
    """
    alpha, beta = map(float, belief.validity["optimize"])
    if not np.isfinite([alpha, beta]).all() or min(alpha, beta) <= 0:
        raise ValueError("Optimize-only planning needs positive finite Beta validity counts")
    if not math.isclose(ph, alpha / (alpha + beta), rel_tol=1e-12, abs_tol=1e-14):
        raise ValueError("Optimization validity probability disagrees with its Beta counts")
    p_after_valid = (alpha + 1) / (alpha + beta + 1)
    p_after_invalid = alpha / (alpha + beta + 1)
    rows = []
    for first in h_indices:
        second = [i for i in h_indices if i != first]
        valid_continuation = invalid_continuation = 0.0
        rejection_probability = None
        effective_depth = 2 if remaining >= 2 and second else 1
        if effective_depth == 2:
            channel_sd = []
            quality_mean = 0.0
            for channel, weight in zip(cfg["channels"], weights):
                mu, covariance = high[channel]
                mu, covariance = np.asarray(mu, dtype=float), np.asarray(covariance, dtype=float)
                if (
                    mu.shape != (len(pool),)
                    or covariance.shape != (len(pool), len(pool))
                    or not np.isfinite(mu).all()
                    or not np.isfinite(covariance).all()
                ):
                    raise ValueError("Invalid high-fidelity Gaussian response moments")
                variance = float(covariance[first, first])
                if variance < -1e-9:
                    raise ValueError("Negative predictive variance")
                channel_sd.append(math.sqrt(max(variance, 0.0)))
                quality_mean += weight * mu[first]
            rejected_normals, rejection_probability = _rejection_normals(
                normals,
                weights * channel_sd,
                incumbent + cfg["acceptance_tolerance"] - quality_mean,
            )
            conditional_mean = np.zeros((len(normals), len(pool)))
            conditional_variance = np.zeros(len(pool))
            for ci, (channel, weight) in enumerate(zip(cfg["channels"], weights)):
                if rejection_probability == 0:
                    break
                mu, covariance = high[channel]
                mu, covariance = np.asarray(mu, dtype=float), np.asarray(covariance, dtype=float)
                sd = channel_sd[ci]
                if sd > 0:
                    # Work with covariance/sd, avoiding an unnecessary division
                    # by a tiny variance before multiplying by the innovation.
                    regression = covariance[:, first] / sd
                    shifted = mu[None, :] + rejected_normals[:, ci, None] * regression[None, :]
                    variance_after = np.diag(covariance) - regression**2
                else:
                    shifted = np.broadcast_to(mu, conditional_mean.shape)
                    variance_after = np.diag(covariance)
                if np.min(variance_after) < -1e-9:
                    raise RuntimeError("Negative optimize-only fantasy conditional variance")
                conditional_mean += weight * shifted
                conditional_variance += weight**2 * np.maximum(variance_after, 0.0)
            if rejection_probability > 0:
                gains = p_after_valid * expected_improvement(
                    conditional_mean[:, second],
                    conditional_variance[second],
                    incumbent,
                    cfg["acceptance_tolerance"],
                )
                valid_continuation = ph * rejection_probability * float(np.mean(gains.max(axis=1)))
            # Invalid attempts reveal no quality; only their Beta failure count
            # changes. direct/ph recovers the unchanged conditional quality EI.
            invalid_continuation = (1 - ph) * p_after_invalid * float(np.max(direct[second])) / ph
        immediate = float(direct[first])
        rows.append(
            dict(
                active=list(pool[first]),
                value=immediate + valid_continuation + invalid_continuation,
                immediate_valid_expected_gain=immediate,
                valid_rejection_continuation=valid_continuation,
                invalid_continuation=invalid_continuation,
                eligible=True,
                effective_depth=effective_depth,
                second_candidates=[list(pool[i]) for i in second] if effective_depth == 2 else [],
                valid_rejection_probability=rejection_probability,
                rejection_probability_method="analytic_gaussian_cdf"
                if effective_depth == 2
                else None,
            )
        )
    best_value = max(row["value"] for row in rows)
    chosen = min(
        (row for row in rows if best_value - row["value"] <= cfg["voi_tolerance"]),
        key=lambda row: row["active"],
    )
    best_immediate = max(float(direct[i]) for i in h_indices)
    immediate_choice = min(
        (pool[i] for i in h_indices if best_immediate - direct[i] <= cfg["voi_tolerance"])
    )
    return dict(
        kind="optimize",
        active=chosen["active"],
        pool=[list(a) for a in pool],
        audit=dict(
            generation=generation,
            seed=seed,
            fantasies=cfg["fantasies"],
            reason="two_step_optimization_value"
            if chosen["effective_depth"] == 2
            else "one_step_expected_improvement",
            planner_policy=policy,
            planner_policy_source=policy_source,
            objective="expected_accepted_quality_gain_until_first_acceptance_within_two_optimizations_v1",
            lookahead="at_most_two_optimizations_until_first_acceptance",
            terminal_expectation="analytic_gaussian_thresholded_expected_improvement",
            continuation_estimator="analytic_rejection_probability_times_conditional_gaussian_fantasies",
            fixed_pool_during_fantasies=True,
            accepted_branch_continuation=0.0,
            unknown_output_basis_simulated=False,
            online_cost_learning=False,
            optimize_values=rows,
            probe_values=[],
            excluded_optimizations=[list(pool[i]) for i in range(len(pool)) if i not in h_indices],
            one_step_choice=list(immediate_choice),
            differs_from_one_step=(chosen["active"] != list(immediate_choice)),
            success_probabilities=dict(
                optimize=ph, after_valid=p_after_valid, after_invalid=p_after_invalid
            ),
            optimize_validity_beta=[alpha, beta],
            quality_prior_independent_channels=True,
            model_early_stop=False,
            acceptance_tolerance=cfg["acceptance_tolerance"],
            voi_tolerance=cfg["voi_tolerance"],
            jitter=belief.jitter_used.copy(),
        ),
    )


def choose(
    context,
    belief,
    incumbent_quality,
    remaining_diagnostics,
    remaining_optimizations,
    config,
    event_index=0,
    excluded=None,
    survey=None,
    *,
    planner_policy=None,
):
    cfg = configuration(config)
    policy = fixed_planner_policy(planner_policy)
    if cfg.get("survey_model") == "posterior_evidence_capture_v1" and policy != dict(
        schema="pas_optimize_only_two_step_v1", depth=2
    ):
        raise ValueError(
            "Posterior capture requires optimize-only H2 without a first-action override"
        )
    policy_source = "source_default" if planner_policy is None else "explicit_frozen_policy"
    if remaining_diagnostics < 0 or remaining_optimizations < 0:
        raise ValueError("Remaining budgets cannot be negative")
    if not np.isfinite(incumbent_quality) or not 0 <= incumbent_quality <= 1:
        raise ValueError("Incumbent quality must be a real normalized physical metric")
    if remaining_optimizations == 0:
        return dict(
            kind="stop",
            reason="optimization_budget_exhausted",
            active=None,
            pool=[],
            audit=dict(
                reason="optimization_budget_exhausted",
                scientific_convergence=False,
                diagnostics_not_purchased=True,
                planner_policy=policy,
                planner_policy_source=policy_source,
            ),
        )
    if "input_response_model" in cfg:
        if (
            survey is None
            or not survey.get("valid")
            or survey.get("context_id") != context.context_id
        ):
            raise ValueError("Input response requires the current bound survey before planning")
        belief.bind_input_evidence(context, survey["proposal_quadratic"])
    excluded = {
        kind: {tuple(sorted(a)) for a in (excluded or {}).get(kind, [])}
        for kind in ("probe", "optimize")
    }
    pool, generation = propose(context, belief, cfg, event_index, excluded, survey)
    if "density_response_model" in cfg:
        generation["density_response"] = belief.density_response_audit(pool)
    if "input_response_model" in cfg and pool:
        generation["input_response"] = belief.input_response_audit(pool)
    h_indices = [i for i, active in enumerate(pool) if active not in excluded["optimize"]]
    if not h_indices:
        # This is a proposal failure, never a model-based convergence certificate.
        return dict(
            kind="stop",
            reason="no_unattempted_optimization_proposal",
            active=None,
            pool=[list(a) for a in pool],
            audit=dict(
                reason="no_unattempted_optimization_proposal",
                scientific_convergence=False,
                generation=generation,
                planner_policy=policy,
                planner_policy_source=policy_source,
            ),
        )
    if (
        policy["schema"] == "pas_survey_first_then_two_step_v1"
        and survey is not None
        and not excluded["probe"]
        and not excluded["optimize"]
    ):
        # Input evidence guides one exploration action in this context. It is
        # not an observation or an expected output quality in the response GP.
        fields = survey.get("proposal_quadratic")
        if fields is None:
            raise ValueError("Survey-first policy requires a numeric proposal_quadratic")
        objective = Quadratic(
            float(fields.get("constant", 0)),
            np.asarray(fields["linear"], dtype=float),
            np.asarray(fields["quadratic"], dtype=float),
        )
        rows = [dict(active=list(pool[i]), score=objective(pool[i])) for i in h_indices]
        if not all(math.isfinite(row["score"]) for row in rows):
            raise ValueError("Survey-first proposal scores must be finite")
        chosen = min(rows, key=lambda row: (-row["score"], row["active"]))
        return dict(
            kind="optimize",
            active=chosen["active"],
            pool=[list(a) for a in pool],
            audit=dict(
                generation=generation,
                reason="evidence_directed_exploration",
                planner_policy=policy,
                planner_policy_source=policy_source,
                objective="survey_input_quadratic_argmax_within_proposed_pool",
                not_expected_quality=True,
                unchanged_true_learning=True,
                survey_input_scope=dict(
                    context_id=survey["context_id"],
                    evidence_id=survey.get("evidence_id"),
                    active=survey.get("active"),
                    scope=fields.get("scope"),
                ),
                survey_values=rows,
                optimize_values=[],
                probe_values=[],
                tie_break="exact_score_then_lexicographic_active",
                reward_voi_tolerance_applied=False,
                model_early_stop=False,
                subsequent_policy="pas_optimize_only_two_step_v1",
            ),
        )
    seed = derived_seed(cfg["seed"], seed_context(context, cfg), event_index, "fantasies")
    rng = np.random.default_rng(seed)
    half = rng.standard_normal((cfg["fantasies"] // 2, len(cfg["channels"])))
    normals = np.concatenate([half, -half], axis=0)
    direct, high, weights, ph, pl = action_values(belief, pool, incumbent_quality, cfg, normals)
    if policy["schema"] in {"pas_optimize_only_two_step_v1", "pas_survey_first_then_two_step_v1"}:
        return _choose_optimize_only(
            belief,
            pool,
            h_indices,
            direct,
            high,
            weights,
            ph,
            incumbent_quality,
            remaining_optimizations,
            cfg,
            normals,
            policy,
            policy_source,
            generation,
            seed,
        )
    ratio = policy["fixed_cost_ratio"]
    time_factor = ratio / (1.0 + ratio)
    best_high_column = int(np.argmax(direct[h_indices]))
    best_direct = float(direct[h_indices[best_high_column]])
    # Stable lexicographic orbital IDs resolve near-ties only after action kind.
    high_ties = [i for i in h_indices if best_direct - direct[i] <= cfg["voi_tolerance"]]
    chosen_index = min(high_ties, key=lambda i: pool[i])
    selected_kind = "optimize"
    values = []
    # A fixed-orbital composition calculation observes an already known value.
    # Do not even consider purchasing it (or rely on floating-point VOI == 0).
    probe_index = None
    informative_probe = bool(set(cfg["channels"]) & {"occupation", "qit"})
    reason = "direct_optimization_rate"
    if remaining_diagnostics and informative_probe:
        for d, active in enumerate(pool):
            if active in excluded["probe"]:
                continue
            fantasy_mean = np.zeros((cfg["fantasies"], len(pool)))
            fantasy_variance = np.zeros(len(pool))
            for ci, (channel, weight) in enumerate(zip(cfg["channels"], weights)):
                means, cov = belief.predict_mixed(
                    pool + [active], ["optimize"] * len(pool) + ["probe"], channel
                )
                mu, hcov = high[channel]
                low_variance = max(float(cov[-1, -1]), 0.0)
                cross = cov[:-1, -1]
                if low_variance > 1e-14:
                    shifted = mu[None, :] + normals[:, ci, None] * cross[None, :] / math.sqrt(
                        low_variance
                    )
                    var = np.diag(hcov) - cross**2 / low_variance
                else:
                    shifted = np.broadcast_to(mu, fantasy_mean.shape)
                    var = np.diag(hcov)
                if np.min(var) < -1e-9:
                    raise RuntimeError("Negative fantasy conditional variance")
                fantasy_mean += weight * shifted
                fantasy_variance += weight**2 * np.maximum(var, 0)
            gains = ph * expected_improvement(
                fantasy_mean[:, h_indices],
                fantasy_variance[h_indices],
                incumbent_quality,
                cfg["acceptance_tolerance"],
            )
            # Tower identity: E[G_best_before(Y)] == best_direct. Subtract the
            # same-sample baseline so its MC error cannot masquerade as learning.
            # Use the true maximizer, before the near-tie dispatch convention.
            knowledge_gain = pl * float((gains.max(axis=1) - gains[:, best_high_column]).mean())
            information_value = best_direct + knowledge_gain
            # Rates use one nominal optimization cost as the common time unit:
            # H costs 1; probe-then-H costs (1 + 1/ratio). No online timing fit.
            value = time_factor * information_value
            values.append(
                dict(
                    active=list(active),
                    value=value,
                    knowledge_gain=knowledge_gain,
                    information_only_value=information_value,
                    time_adjusted_value=value,
                    time_adjusted_knowledge_gain=value - best_direct,
                    return_rate=value,
                )
            )
        eligible_probes = [
            row for row in values if row["value"] > best_direct + cfg["voi_tolerance"]
        ]
        if eligible_probes:
            best_probe = max(row["value"] for row in eligible_probes)
            tied = [
                tuple(row["active"])
                for row in eligible_probes
                if best_probe - row["value"] <= cfg["voi_tolerance"]
            ]
            probe_index = pool.index(min(tied))
        if probe_index is not None:
            selected_kind, chosen_index = "probe", probe_index
            reason = "time_adjusted_information_gain"
    return dict(
        kind=selected_kind,
        active=list(pool[chosen_index]),
        pool=[list(a) for a in pool],
        audit=dict(
            generation=generation,
            seed=seed,
            fantasies=cfg["fantasies"],
            terminal_expectation="analytic_gaussian_thresholded_expected_improvement",
            lookahead="at_most_one_final_optimization",
            fixed_pool_during_fantasies=True,
            information_value_estimator="same_sample_terminal_advantage",
            reason=reason,
            planner_policy=policy,
            planner_policy_source=policy_source,
            objective="expected_quality_gain_per_fixed_relative_cost_v1",
            return_rate_unit="quality_per_nominal_optimization_cost",
            time_adjustment_factor=time_factor,
            online_cost_learning=False,
            optimize_values=[
                dict(
                    active=list(a),
                    value=float(v),
                    eligible=i in h_indices,
                    information_only_value=float(v),
                    time_adjusted_value=float(v),
                    return_rate=float(v),
                )
                for i, (a, v) in enumerate(zip(pool, direct))
            ],
            probe_values=values,
            success_probabilities=dict(probe=pl, optimize=ph),
            diagnostic_has_unknown_quality=informative_probe,
            quality_prior_independent_channels=True,
            model_early_stop=False,
            acceptance_tolerance=cfg["acceptance_tolerance"],
            voi_tolerance=cfg["voi_tolerance"],
            jitter=belief.jitter_used.copy(),
        ),
    )
