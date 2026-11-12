"""Measured terminal utilities; unavailable enabled evidence is never zero."""

from __future__ import annotations

import math

import numpy as np

PAIR_COMPOSITION_MODELS = ("determinant_pair_v1", "harmonic_determinant_pair_v1")
HARMONIC_CHANNEL_MODEL = "harmonic_channel_pair_v1"
COMPLEMENTARITY_COMPOSITION_MODELS = PAIR_COMPOSITION_MODELS + (HARMONIC_CHANNEL_MODEL,)


def validate_composition_purity_weight(value, model="determinant_pair_v1"):
    """The arithmetic purity/coverage tradeoff is frozen with the run."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise ValueError("composition_purity_weight must be a finite number in [0, 1]")
    if value != 0.5 and model != "determinant_pair_v1":
        raise ValueError("Non-default composition_purity_weight requires determinant_pair_v1")
    return float(value)


def _weights(values, count):
    weights = np.asarray(values, float)
    if (
        weights.shape != (count,)
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0)
        or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError("State weights must be finite, nonnegative and normalized")
    return weights


def validate_densities(densities, state_weights, electrons=None):
    d = np.asarray(densities)
    if d.ndim != 3 or d.shape[1] != d.shape[2] or not d.shape[1] or not np.all(np.isfinite(d)):
        raise ValueError("Expected complete finite root x K x K one-particle densities")
    _weights(state_weights, len(d))
    if not np.allclose(d, d.swapaxes(1, 2).conj(), atol=1e-7, rtol=0):
        raise ValueError("One-particle density is not Hermitian")
    eigenvalues = np.linalg.eigvalsh(d)
    if eigenvalues.min() < -1e-7 or eigenvalues.max() > 2 + 1e-7:
        raise ValueError("Spin-free density eigenvalues outside [0, 2]")
    if electrons is not None and not np.allclose(
        np.trace(d, axis1=1, axis2=2), electrons, atol=1e-6, rtol=0
    ):
        raise ValueError("State density electron trace differs from its calculation scope")
    return d


def occupation(densities, state_weights, electrons=None):
    d = validate_densities(densities, state_weights, electrons)
    values = (2 * np.trace(d, axis1=1, axis2=2) - np.einsum("rij,rji->r", d, d)).real / d.shape[1]
    return float(np.clip(_weights(state_weights, len(d)) @ values, 0.0, 1.0))


def validate_composition_pair(pair, nmo):
    """Validate the full-matrix x M x convention for determinant coverage."""
    matrix = np.asarray(pair, float)
    if (
        matrix.shape != (nmo, nmo)
        or not np.all(np.isfinite(matrix))
        or np.any(matrix < 0)
        or np.any(np.diag(matrix) != 0)
        or not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0)
    ):
        raise ValueError(
            "Composition pair matrix must be finite, nonnegative, symmetric and zero-diagonal"
        )
    return matrix


def validate_composition_channels(channels, nmo):
    """Validate full-basis squared channel amplitudes and normalized weights."""
    if (
        not isinstance(channels, dict)
        or set(channels) != {"schema", "plus", "minus", "weights"}
        or channels["schema"] != "target_ao_harmonic_channels_v1"
    ):
        raise ValueError("Invalid composition channel schema")
    plus, minus, weights = (
        np.asarray(channels[key], float) for key in ("plus", "minus", "weights")
    )
    if (
        plus.ndim != 2
        or plus.shape[1] != nmo
        or not len(plus)
        or minus.shape != plus.shape
        or weights.shape != (len(plus),)
        or not all(np.all(np.isfinite(a)) for a in (plus, minus, weights))
        or np.any(plus < 0)
        or np.any(minus < 0)
        or np.any(weights <= 0)
        or np.any(plus > 1 + 1e-7)
        or np.any(minus > 1 + 1e-7)
        or not np.allclose(plus.sum(axis=1), 1.0, atol=1e-7, rtol=0)
        or not np.allclose(minus.sum(axis=1), 1.0, atol=1e-7, rtol=0)
        or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0)
    ):
        raise ValueError(
            "Composition channels require finite normalized full-basis projections and weights"
        )
    return dict(
        schema=channels["schema"],
        plus=plus.tolist(),
        minus=minus.tolist(),
        weights=weights.tolist(),
    )


def _harmonic(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    # Ratios before products avoid underflow for small but nonzero evidence.
    ratio = np.divide(np.maximum(a, b), a + b, out=np.zeros_like(a + b), where=a + b > 0)
    return 2 * np.minimum(a, b) * ratio


def _harmonic_slopes(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    left = np.divide(b, a + b, out=np.zeros_like(a + b), where=a + b > 0)
    right = np.divide(a, a + b, out=np.zeros_like(a + b), where=a + b > 0)
    return np.where(a + b > 0, 2 * left**2, 0.5), np.where(a + b > 0, 2 * right**2, 0.5)


def _composition_inputs(
    projection, active, composition_pair, composition_model, composition_channels
):
    projection = np.asarray(projection, float)
    active = np.asarray(active, int)
    if (
        projection.ndim != 1
        or not np.all(np.isfinite(projection))
        or np.any(projection < -1e-8)
        or np.any(projection > 1 + 1e-8)
    ):
        raise ValueError("Invalid metric AO-target projections")
    if (
        active.ndim != 1
        or not len(active)
        or len(set(active)) != len(active)
        or np.any(active < 0)
        or np.any(active >= len(projection))
    ):
        raise ValueError("Invalid active indices for composition")
    if composition_model == "projection":
        if composition_pair is not None or composition_channels is not None:
            raise ValueError("Projection composition model cannot consume pair or channel data")
        return projection, active, None, None
    if composition_model == HARMONIC_CHANNEL_MODEL:
        if composition_pair is not None or composition_channels is None:
            raise ValueError(
                "Harmonic channel composition requires channel data and no determinant matrix"
            )
        channels = validate_composition_channels(composition_channels, len(projection))
        return (
            projection,
            active,
            None,
            {key: np.asarray(channels[key], float) for key in ("plus", "minus", "weights")},
        )
    if composition_model not in PAIR_COMPOSITION_MODELS:
        raise ValueError("Unknown composition model")
    if composition_channels is not None:
        raise ValueError("Determinant composition cannot consume harmonic channel data")
    if composition_pair is None:
        raise ValueError("Enabled determinant composition requires its pair matrix")
    pair = validate_composition_pair(composition_pair, len(projection))
    return projection, active, pair, None


def _composition_coverage(active, pair, channels):
    if channels is None:
        coverage = float(pair[np.ix_(active, active)].sum())
    else:
        plus = channels["plus"][:, active].sum(axis=1)
        minus = channels["minus"][:, active].sum(axis=1)
        coverage = float(channels["weights"] @ _harmonic(plus, minus))
    if not -1e-8 <= coverage <= 1 + 1e-8:
        raise ValueError("Composition pair coverage is outside [0, 1]")
    return float(np.clip(coverage, 0.0, 1.0))


def _composition_normalizers(projection, active, pair, channels, capture, window):
    if not capture:
        return float(len(active)), 1.0
    if window is None:
        raise ValueError("Composition capture requires the admissible window")
    window = np.asarray(window, int)
    if (
        window.ndim != 1
        or not len(window)
        or len(set(window)) != len(window)
        or window.min() < 0
        or window.max() >= len(projection)
        or not set(active) <= set(window)
    ):
        raise ValueError("Invalid composition capture window")
    pnorm = float(projection[window].sum())
    bnorm = (
        _composition_coverage(window, pair, channels)
        if pair is not None or channels is not None
        else 1.0
    )
    if pnorm <= 0 or bnorm <= 0:
        raise ValueError("No supported composition evidence in capture window")
    return pnorm, bnorm


def composition_components(
    projection,
    active,
    composition_pair=None,
    composition_model="projection",
    *,
    composition_channels=None,
    capture=False,
    window=None,
):
    """Exact terminal components, or legacy window captures for diagnostics.

    Capture rescales P and B separately against W; it is a distinct input
    objective, not a proportional rescaling of the raw terminal objective.
    Live composition proposals use the default terminal scale, not capture.
    """
    projection, active, pair, channels = _composition_inputs(
        projection, active, composition_pair, composition_model, composition_channels
    )
    pnorm, bnorm = _composition_normalizers(projection, active, pair, channels, capture, window)
    p = float(np.clip(projection[active].sum() / pnorm, 0.0, 1.0))
    b = (
        None
        if pair is None and channels is None
        else _composition_coverage(active, pair, channels) / bnorm
    )
    return p, b


def composition_score(
    projection,
    active,
    composition_pair=None,
    composition_model="projection",
    *,
    composition_channels=None,
    capture=False,
    window=None,
    composition_purity_weight=0.5,
    composition_preferences=None,
):
    if composition_preferences is not None and capture:
        raise ValueError("Composition preferences require the terminal scale, not window capture")
    alpha = validate_composition_purity_weight(composition_purity_weight, composition_model)
    p, b = composition_components(
        projection,
        active,
        composition_pair,
        composition_model,
        composition_channels=composition_channels,
        capture=capture,
        window=window,
    )
    if b is None:
        value = p
    elif composition_model in ("harmonic_determinant_pair_v1", HARMONIC_CHANNEL_MODEL):
        # Continuous extension at the origin; no epsilon changes the objective.
        value = 0.0 if p + b == 0.0 else 2.0 * min(p, b) * (max(p, b) / (p + b))
    else:
        value = 0.5 * (p + b) if alpha == 0.5 else alpha * p + (1 - alpha) * b
    if composition_preferences is not None:
        from .preferences import preference_score

        value, _ = preference_score(value, composition_preferences, active, len(projection))
    return value


def composition_proposal_weights(
    projection,
    active,
    composition_pair=None,
    composition_model="projection",
    *,
    composition_channels=None,
    capture=False,
    window=None,
    composition_purity_weight=0.5,
):
    """Coefficients of the proposal-only plane anchored at the accepted active.

    Harmonic H is concave in (P, B), so its tangent is an upper bound, not an
    equivalent objective or an improvement guarantee. At the nondifferentiable
    origin use the symmetric supporting plane (P+B)/2, which also touches H=0.
    """
    alpha = validate_composition_purity_weight(composition_purity_weight, composition_model)
    p, b = composition_components(
        projection,
        active,
        composition_pair,
        composition_model,
        composition_channels=composition_channels,
        capture=capture,
        window=window,
    )
    if b is None:
        return 1.0, 0.0
    if (
        composition_model in ("harmonic_determinant_pair_v1", HARMONIC_CHANNEL_MODEL)
        and p + b > 0.0
    ):
        return 2.0 * (b / (p + b)) ** 2, 2.0 * (p / (p + b)) ** 2
    return alpha, 1 - alpha


def composition_quadratic(
    projection,
    anchor,
    composition_pair=None,
    composition_model="projection",
    *,
    composition_channels=None,
    capture=False,
    window=None,
    composition_purity_weight=0.5,
    composition_preferences=None,
):
    """Return constant + linear@x + x@pair@x in the chosen component scale.

    Arithmetic determinant is exact at fixed K. For harmonic channels, inner
    and outer supporting tangents touch the exact score at the incumbent.
    Their nested composition is an upper bound, not an improvement guarantee.
    """
    projection, active, pair, channels = _composition_inputs(
        projection, anchor, composition_pair, composition_model, composition_channels
    )
    if composition_preferences is not None and capture:
        raise ValueError("Composition preferences require the terminal scale, not window capture")
    pnorm, bnorm = _composition_normalizers(projection, active, pair, channels, capture, window)
    a, b = composition_proposal_weights(
        projection,
        active,
        composition_pair,
        composition_model,
        composition_channels=composition_channels,
        capture=capture,
        window=window,
        composition_purity_weight=composition_purity_weight,
    )
    linear = a * projection / pnorm
    quadratic = np.zeros((len(projection), len(projection)))
    if pair is not None:
        quadratic = b * pair / bnorm
    elif channels is not None:
        plus = channels["plus"][:, active].sum(axis=1)
        minus = channels["minus"][:, active].sum(axis=1)
        left, right = _harmonic_slopes(plus, minus)
        linear = (
            linear
            + b
            * (
                channels["weights"]
                @ (left[:, None] * channels["plus"] + right[:, None] * channels["minus"])
            )
            / bnorm
        )
    if composition_preferences is not None:
        from .preferences import preference_quadratic

        extra_linear, extra_pair, scale = preference_quadratic(
            composition_preferences, len(projection), len(active)
        )
        linear, quadratic = (linear + extra_linear) / scale, (quadratic + extra_pair) / scale
    return dict(constant=0.0, linear=linear, pair=quadratic)


def qit_normalizers(size):
    """Denominators for entropy sums and unordered half-MI sums on K orbitals."""
    if not isinstance(size, (int, np.integer)) or isinstance(size, (bool, np.bool_)) or size < 1:
        raise ValueError("QIT requires a positive integer orbital count")
    if size == 1:
        return math.log(4), None
    return 2 * math.log(4) * size, 2 * math.log(4) * (size * (size - 1) / 2)


def qit(entropies, mutual_information, state_weights):
    s = np.asarray(entropies, float)
    mi = np.asarray(mutual_information, float)
    if s.ndim != 2 or not s.shape[1] or mi.shape != (s.shape[0], s.shape[1], s.shape[1]):
        raise ValueError("Complete per-root entropies and mutual information are required")
    if (
        not np.all(np.isfinite(s))
        or not np.all(np.isfinite(mi))
        or np.min(s) < -1e-8
        or np.min(mi) < -1e-8
        or np.max(s) > math.log(4) + 1e-8
        or np.max(mi) > math.log(4) + 1e-8
    ):
        raise ValueError("Invalid QIT range; use natural logs and half-mutual-information")
    if not np.allclose(mi, mi.swapaxes(1, 2), atol=1e-8, rtol=0) or not np.allclose(
        np.diagonal(mi, axis1=1, axis2=2), 0, atol=1e-8, rtol=0
    ):
        raise ValueError("Mutual information must be symmetric with zero diagonal")
    values = s.mean(axis=1) / math.log(4)
    if s.shape[1] > 1:
        pair = mi[:, np.triu_indices(s.shape[1], 1)[0], np.triu_indices(s.shape[1], 1)[1]].mean(
            axis=1
        ) / math.log(4)
        values = 0.5 * (values + pair)
    return float(np.clip(_weights(state_weights, len(s)) @ values, 0.0, 1.0))


def evaluate(
    channels,
    weights,
    *,
    densities=None,
    state_weights=None,
    electrons=None,
    composition=None,
    active=None,
    entropies=None,
    mutual_information=None,
    composition_pair=None,
    composition_model="projection",
    composition_channels=None,
    composition_purity_weight=0.5,
    composition_preferences=None,
):
    channels = tuple(channels)
    if (
        not channels
        or len(set(channels)) != len(channels)
        or not set(channels) <= {"occupation", "composition", "qit"}
    ):
        raise ValueError("Invalid nonempty enabled channel mask")
    if composition_preferences is not None and "composition" not in channels:
        raise ValueError("Composition preferences require the composition channel")
    if set(weights) != set(channels) or any(not np.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("Every enabled channel requires a positive frozen weight")
    result = {}
    for channel in channels:
        if channel == "occupation":
            if densities is None:
                raise ValueError("Enabled occupation evidence is unavailable")
            result[channel] = occupation(densities, state_weights, electrons)
        elif channel == "composition":
            if composition is None or active is None:
                raise ValueError("Enabled composition evidence is unavailable")
            result[channel] = composition_score(
                composition,
                active,
                composition_pair,
                composition_model,
                composition_channels=composition_channels,
                composition_purity_weight=composition_purity_weight,
            )
            if composition_preferences is not None:
                from .preferences import preference_score

                result[channel], result["composition_preferences"] = preference_score(
                    result[channel], composition_preferences, active, len(composition)
                )
        else:
            if entropies is None or mutual_information is None:
                raise ValueError("Enabled QIT evidence is unavailable")
            result[channel] = qit(entropies, mutual_information, state_weights)
    result["quality"] = sum(weights[c] * result[c] for c in channels) / sum(weights.values())
    return result
