# Selected from c06614a test_model.py; retention rationale: docs/maintenance/retention.json
"""Synthetic checks: no molecular solver or remote calculation is invoked."""

import math
from itertools import combinations

import numpy as np
import pytest

from pas.belief import Quadratic, ResponseBelief, kernel
from pas.config import configuration
from pas.contracts import Context
from pas.planner import expected_improvement, fixed_planner_policy
from pas.qubo import solve


def context(n=6, k=2):
    return Context(
        basis_id="basis",
        geometry_id="geometry",
        backend="cas",
        nmo=n,
        active=tuple(range(2, 2 + k)),
        inactive=(0, 1),
        total_electrons=2 * k + 4,
        electrons=2 * k,
        composition=np.linspace(0.1, 0.9, n).tolist(),
        protocol_id="frozen",
    )


def receipt(ctx, active, kind, n=0.6, c=None, eid="one", valid=True):
    if c is None:
        c = float(np.mean(np.asarray(ctx.composition)[list(active)]))
    return dict(
        evidence_id=eid,
        request_id="request-" + eid,
        context_id=ctx.context_id,
        kind=kind,
        active=list(active),
        valid=valid,
        converged=valid,
        state_valid=valid,
        metrics=dict(occupation=n, composition=c),
        cpu_seconds=1.0,
        wall_seconds=1.0,
    )


def cheap_probe_policy():
    """Keep pure information-value regressions in the cheap-diagnostic regime."""
    return fixed_planner_policy(dict(schema="pas_fixed_cost_ratio_v1", fixed_cost_ratio=1000000.0))


def test_zero_noise_batch_posterior_and_known_composition():
    ctx, cfg = (context(), configuration())
    b = ResponseBelief(ctx, cfg)
    train = [(0, 2), (3, 5)]
    values = np.array([0.2, 0.8])
    b.observe(receipt(ctx, train[0], "probe", n=values[0]))
    b.observe(receipt(ctx, train[1], "optimize", n=values[1], eid="two"))
    query = [(0, 2), (1, 4), (3, 5)]
    mean, cov = b.predict(query, "optimize", "occupation")
    ktt = cfg["prior_std"] ** 2 * kernel(train, train, ctx.k)
    ktt[0, 0] += cfg["discrepancy_std"] ** 2
    cross = cfg["prior_std"] ** 2 * kernel(query, train, ctx.k)
    inverse = np.linalg.inv(ktt + cfg["jitter"] * np.eye(2))
    expected = cfg["prior_mean"] + cross @ inverse @ (values - cfg["prior_mean"])
    variance = cfg["prior_std"] ** 2 * kernel(query, query, ctx.k) - cross @ inverse @ cross.T
    np.testing.assert_allclose(mean, expected, atol=1e-12)
    np.testing.assert_allclose(cov, variance, atol=1e-12)
    assert abs(mean[-1] - 0.8) < 1e-09
    assert cov[-1, -1] < 2e-12
    cm, cc = b.predict(query, "probe", "composition")
    np.testing.assert_allclose(cm, [b.composition(a) for a in query])
    assert not np.any(cc)
    assert len(b.observations["composition"]) == 1


def test_quadratic_mean_and_matheron_covariance():
    ctx, cfg = (context(), configuration())
    b = ResponseBelief(ctx, cfg)
    b.observe(receipt(ctx, (0, 2), "probe", n=0.3))
    b.observe(receipt(ctx, (3, 5), "optimize", n=0.75, c=0.6, eid="two"))
    all_active = list(combinations(ctx.window, ctx.k))
    for channel in cfg["channels"]:
        mean, _ = b.predict(all_active, "optimize", channel)
        q = b.posterior_quadratic(channel)
        np.testing.assert_allclose([q(a) for a in all_active], mean, atol=1e-12)
    query = [(0, 1), (1, 4), (3, 5)]
    mean, covariance = b.predict(query, "optimize", "occupation")
    rng = np.random.default_rng(921)
    samples = np.asarray(
        [
            [q(a) for a in query]
            for q in (b.posterior_quadratic("occupation", rng) for _ in range(5000))
        ]
    )
    np.testing.assert_allclose(samples.mean(axis=0), mean, atol=0.009)
    np.testing.assert_allclose(np.cov(samples.T), covariance, atol=0.004)


def test_fantasy_covariance_matches_batch_conditioning():
    ctx, cfg = (context(), configuration())
    query = [(0, 1), (1, 4), (3, 5)]
    b = ResponseBelief(ctx, cfg)
    means, covariance = b.predict_mixed(
        query + [(2, 4)], ["optimize"] * 3 + ["probe"], "occupation"
    )
    observed = 0.73
    gain = covariance[:-1, -1] / (covariance[-1, -1] + cfg["jitter"])
    expected_mean = means[:-1] + gain * (observed - means[-1])
    expected_cov = covariance[:-1, :-1] - np.outer(gain, covariance[-1, :-1])
    b.observe(receipt(ctx, (2, 4), "probe", n=observed))
    mean, cov = b.predict(query, "optimize", "occupation")
    np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
    np.testing.assert_allclose(cov, expected_cov, atol=1e-12)


def test_legality_partition_and_exact_solver_no_occupied_role_filter():
    ctx, cfg = (context(), configuration())
    arbitrary = ctx.partition((4, 5))
    assert arbitrary["active"] == [4, 5] and arbitrary["inactive"] == [0, 1]
    moved = ctx.partition((0, 5))
    assert moved["inactive"] == [1, 2]
    with pytest.raises(ValueError):
        ctx.partition((2, 2))
    q = Quadratic(2.0, np.arange(ctx.nmo, dtype=float), np.zeros((ctx.nmo, ctx.nmo)))
    active, audit = solve(ctx, q, cfg, 1)
    assert active == (4, 5)
    assert audit["examined"] == math.comb(6, 2)
    runner_up, _ = solve(ctx, q, cfg, 1, [active])
    assert runner_up == (3, 5)


def test_expected_improvement_formula_and_zero_variance():
    mean, var, baseline = (np.array([0.4, 0.8, 0.6]), np.array([0.04, 0.0, 0.0]), 0.6)
    result = expected_improvement(mean, var, baseline)
    z = (0.4 - 0.6 - 1e-06) / 0.2
    expected = -0.2 * 0.5 * math.erfc(-z / math.sqrt(2)) + 0.2 * math.exp(-z * z / 2) / math.sqrt(
        2 * math.pi
    )
    np.testing.assert_allclose(result, [expected, 0.2, 0.0])


def test_native_swap_delta_includes_pair_interactions():
    ctx = context(12, 4)
    cfg = configuration(dict(exact_limit=1))
    rng = np.random.default_rng(746)
    matrix = rng.normal(size=(ctx.nmo, ctx.nmo))
    matrix = (matrix + matrix.T) / 2
    np.fill_diagonal(matrix, 0)
    q = Quadratic(0.7, rng.normal(size=ctx.nmo), matrix)
    exact = max(combinations(ctx.window, ctx.k), key=q)
    actual, _ = solve(ctx, q, cfg, 32)
    assert actual == exact


def test_gaussian_update_order_independence_and_channel_isolation():
    ctx, cfg = (context(), configuration())
    rows = [
        receipt(ctx, (0, 2), "probe", n=0.23),
        receipt(ctx, (1, 4), "optimize", n=0.69, c=0.62, eid="two"),
        receipt(ctx, (3, 5), "probe", n=0.74, eid="three"),
    ]
    left, right = (ResponseBelief(ctx, cfg), ResponseBelief(ctx, cfg))
    for row in rows:
        left.observe(row)
    for row in reversed(rows):
        right.observe(row)
    query = list(combinations(ctx.window, ctx.k))
    for channel in cfg["channels"]:
        lm, lc = left.predict(query, "optimize", channel)
        rm, rc = right.predict(query, "optimize", channel)
        np.testing.assert_allclose(lm, rm, atol=1e-12)
        np.testing.assert_allclose(lc, rc, atol=1e-12)
    n_only = ResponseBelief(ctx, configuration(dict(channels=["occupation"])))
    bad_disabled = dict(rows[0], metrics=dict(occupation=0.23, composition=9.0))
    n_only.observe(bad_disabled)
    assert "composition" not in n_only.observations
    with pytest.raises(ValueError, match="Disabled"):
        n_only.predict(query, "optimize", "composition")
