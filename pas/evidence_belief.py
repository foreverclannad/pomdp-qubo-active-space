"""An explicitly uncalibrated Gaussian filter for capture statistics.

The random variables are u_i, b_ij, normalized entropy and normalized MI, not
entries of a physical density matrix. Complete physical densities provide the
observations; Gaussian tails are not a physical wavefunction ensemble. One
survey is conditioned once from the frozen prior, so replay cannot count it
again. The receipt owner must bind the actual solver protocol and provenance.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from .common import identity
from .metrics import (
    composition_quadratic,
    composition_score,
    qit,
    qit_normalizers,
    validate_composition_purity_weight,
    validate_densities,
)


def validate_proposal_score_model(fields, config):
    """Never reinterpret archived window-scaled evidence as terminal-aligned."""
    descriptor = fields.get("composition_descriptor") or {}
    from .preferences import normalize_preferences

    preferences = normalize_preferences(config.get("composition_preferences"))
    if descriptor.get("preferences_id") != (
        identity(preferences) if preferences is not None else None
    ):
        raise ValueError("Proposal composition preferences differ from the frozen configuration")
    if descriptor.get("composition_purity_weight", 0.5) != config.get(
        "composition_purity_weight", 0.5
    ):
        raise ValueError("Proposal composition purity weight differs from the frozen configuration")
    model = config.get("proposal_score_model")
    if fields.get("proposal_score_model") != model:
        raise ValueError("Proposal scoring model differs from the frozen configuration")
    if model is not None:
        if model != "terminal_aligned_v1":
            raise ValueError("Unknown proposal scoring model")
        if (fields.get("composition_descriptor") or {}).get("model", "projection") != config[
            "composition_model"
        ]:
            raise ValueError("Proposal composition differs from the frozen scoring model")
        has_qit = "qit" in config["channels"]
        if has_qit and (
            fields.get("qit_scale") != "terminal_K_pair_count_ln4"
            or fields.get("cut_penalty") != 1.0
        ):
            raise ValueError(
                "Terminal-aligned QIT requires the bound terminal scale and unit cut penalty"
            )


def _composition_descriptor(context, model, measured, composition_purity_weight=0.5):
    """Bind exact AO data and a terminal-scale, incumbent-anchored proposal."""
    preferences = context.composition_preferences
    if model not in (
        "determinant_pair_v1",
        "harmonic_determinant_pair_v1",
        "harmonic_channel_pair_v1",
    ) and not (model == "projection" and preferences is not None):
        raise ValueError("Unknown complementary composition descriptor")
    channels = getattr(context, "composition_channels", None)
    quadratic = composition_quadratic(
        context.composition,
        context.active,
        context.composition_pair,
        model,
        composition_channels=channels,
        composition_preferences=preferences,
        composition_purity_weight=composition_purity_weight,
    )
    data = dict(projection=context.composition, pair=context.composition_pair, channels=channels)
    if preferences is not None:
        data["preferences"] = preferences
    descriptor = dict(
        schema="pas_capture_composition_v2",
        model=model,
        context_id=context.context_id,
        basis_id=context.basis_id,
        anchor_active=list(context.active),
        measured_indices=list(measured),
        data_id=identity(data),
        capture_normalization="terminal_projection_mean_and_raw_pair_coverage",
        response_scale="exact_terminal_composition_descriptor",
        proposal_exact=model in ("projection", "determinant_pair_v1"),
        proposal_approximation="none"
        if model in ("projection", "determinant_pair_v1")
        else "incumbent_supporting_tangent",
        composition_input_uncertainty=0.0,
        target_quality_observation=False,
    )
    if composition_purity_weight != 0.5:
        descriptor["composition_purity_weight"] = float(composition_purity_weight)
    if preferences is not None:
        descriptor["preferences_id"] = preferences["specification_id"]
        descriptor["preference_normalization"] = (
            "fixed_base_plus_penalty_complements_plus_pair_rewards"
        )
    return descriptor, quadratic


def _finite_scalar(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and np.isfinite(value)
    )


def _capture_channels(backend):
    if backend == "dmrg":
        return ("occupation", "composition", "qit")
    if backend == "cas":
        return ("occupation", "composition")
    raise ValueError("Capture evidence requires a native CAS or DMRG backend")


class EvidenceResponse:
    """Input descriptors on the terminal-metric scale, not target truth.

    Entropy/MI totals are conditioned on their observed values. This does not
    certify finite-M error, infer a target-electron wavefunction, or transport
    QIT through an orbital rotation. A response model must add its own residual
    uncertainty for those differences; Gaussian descriptors are never clipped.
    AO composition is exact and deterministic, including nonlinear pair scores.
    """

    def __init__(self, context, fields):
        self.context = deepcopy(context)
        self.context_id = context.context_id
        self.channels = _capture_channels(context.backend)
        has_qit = "qit" in self.channels
        posterior = fields.get("input_posterior", {})
        if (
            context.k < 2
            or fields.get("score_model") != "posterior_evidence_capture_v1"
            or fields.get("composition_scale") != "terminal_score"
            or fields.get("scope") != "input_capture_statistics_posterior_not_target_CAS_truth"
            or posterior.get("schema") != "pas_capture_statistics_posterior_v1"
        ):
            raise ValueError("Evidence response requires the bound capture-statistics posterior")
        if any(
            fields.get(key) != list(self.channels)
            for key in ("score_channels", "evidence_channels")
        ):
            raise ValueError(
                "Evidence response channels differ from the supported physical evidence"
            )
        for key, expected in (
            ("context_id", context.context_id),
            ("basis_id", context.basis_id),
            ("geometry_id", context.geometry_id),
            ("states", context.states),
        ):
            if posterior.get(key) != expected:
                raise ValueError("Evidence response has a different " + key)
        measured = posterior.get("measured_indices", [])
        if (
            not isinstance(measured, list)
            or len(measured) < context.k
            or any(type(i) is not int for i in measured)
            or len(set(measured)) != len(measured)
            or not set(measured) <= set(context.window)
            or fields.get("measured_indices") != measured
        ):
            raise ValueError("Evidence response has an invalid measured scope")
        count = len(measured)
        electrons = posterior.get("electrons")
        if type(electrons) is not int or not 0 <= electrons <= 2 * count:
            raise ValueError("Evidence response lacks its diagnostic electron scope")
        if (
            posterior.get("pair_order") != "upper_triangle_row_major_in_measured_order"
            or posterior.get("cross_block_covariance") != "assumed_zero"
            or posterior.get("normalized_blocks") != (["h", "m"] if has_qit else [])
        ):
            raise ValueError("Unsupported evidence covariance or pair convention")
        self.measured_indices = tuple(measured)
        self._first, self._second = np.triu_indices(count, 1)
        self._blocks = {}
        blocks = posterior.get("blocks", {})
        block_names = ("u", "b", "h", "m") if has_qit else ("u", "b")
        if set(blocks) != set(block_names):
            raise ValueError("Evidence response statistic blocks differ from its physical channels")
        for name in block_names:
            size = count if name in ("u", "h") else len(self._first)
            normalized = name in ("h", "m")
            block = blocks[name]
            for key in ("prior_mean", "observation", "mean"):
                raw = np.asarray(block.get(key))
                if (
                    raw.dtype.kind not in "iuf"
                    or raw.shape != (size,)
                    or not np.all(np.isfinite(raw))
                ):
                    raise ValueError("Invalid evidence statistic " + name + "." + key)
                if normalized and not np.isclose(raw.sum(), 1.0, atol=1e-12, rtol=0):
                    raise ValueError("Evidence shares are not normalized")
            covariance = block.get("covariance", {})
            kind = "scaled_sum_zero_projector" if normalized else "scaled_identity"
            scale = covariance.get("scale", float("nan"))
            diagonal = covariance.get("diagonal_variance", float("nan"))
            if (
                covariance.get("kind") != kind
                or type(covariance.get("dimension")) is not int
                or covariance["dimension"] != size
                or not _finite_scalar(scale)
                or scale < 0
                or not _finite_scalar(diagonal)
                or diagonal < 0
                or not np.isclose(
                    diagonal, scale * (1 - 1 / size if normalized else 1), atol=1e-14, rtol=1e-12
                )
            ):
                raise ValueError("Invalid compact evidence covariance for " + name)
            self._blocks[name] = (np.array(block["mean"], dtype=float), float(scale), normalized)
        normalization = fields.get("normalization", {})
        self._totals = {}
        for name in ("entropy", "mutual_information") if has_qit else ():
            value = normalization.get(name, float("nan"))
            if not _finite_scalar(value) or value <= 0:
                raise ValueError("Evidence response needs positive finite observed QIT totals")
            self._totals[name] = float(value)
        if not has_qit and any(
            normalization.get(name) is not None for name in ("entropy", "mutual_information")
        ):
            raise ValueError("CAS evidence cannot supply QIT normalization")
        self._proposal_score_model = fields.get("proposal_score_model")
        if self._proposal_score_model not in (None, "terminal_aligned_v1"):
            raise ValueError("Unknown proposal scoring model")
        if (
            self._proposal_score_model is not None
            and has_qit
            and (
                fields.get("qit_scale") != "terminal_K_pair_count_ln4"
                or fields.get("cut_penalty") != 1.0
            )
        ):
            raise ValueError("Terminal-aligned QIT requires its scale and unit cut penalty")
        self._projection = np.asarray(context.composition, dtype=float).copy()
        if (
            self._projection.shape != (context.nmo,)
            or not np.isfinite(self._projection).all()
            or np.any(self._projection < 0)
            or np.any(self._projection > 1)
        ):
            raise ValueError("Evidence response requires exact AO projections")
        descriptor = fields.get("composition_descriptor")
        if context.composition_preferences is not None and descriptor is None:
            raise ValueError("Missing bound composition preference descriptor")
        self._composition_model = "projection"
        self._composition_purity_weight = 0.5
        extra = {}
        if descriptor is not None:
            if not isinstance(descriptor, dict):
                raise ValueError("Malformed complementary composition descriptor")
            self._composition_model = descriptor.get("model")
            self._composition_purity_weight = validate_composition_purity_weight(
                descriptor.get("composition_purity_weight", 0.5), self._composition_model
            )
            expected, _ = _composition_descriptor(
                context, self._composition_model, measured, self._composition_purity_weight
            )
            if descriptor != expected:
                raise ValueError("Composition descriptor differs from its bound physical input")
            extra["composition_descriptor"] = descriptor
        composition_score(
            context.composition,
            context.active,
            context.composition_pair,
            self._composition_model,
            composition_channels=getattr(context, "composition_channels", None),
            composition_preferences=context.composition_preferences,
            composition_purity_weight=self._composition_purity_weight,
        )
        self.evidence_id = identity(dict(posterior=posterior, normalization=normalization, **extra))
        self.audit = dict(
            schema="pas_evidence_response_baseline_v1",
            context_id=self.context_id,
            basis_id=context.basis_id,
            evidence_id=self.evidence_id,
            measured_indices=measured.copy(),
            diagnostic_electrons=electrons,
            target_electrons=context.electrons,
            calibrated=False,
            conditional_on_observed_QIT_totals=has_qit,
            physical_matrix_posterior=False,
            target_quality_observation=False,
            cross_basis_transport=False,
        )
        if extra:
            self.audit.update(extra)

    def _selection(self, active_list):
        active_list = list(active_list)
        selected = np.zeros((len(active_list), len(self.measured_indices)))
        lookup = {orbital: i for i, orbital in enumerate(self.measured_indices)}
        for row, active in enumerate(active_list):
            active = self.context.partition(active)["active"]
            if not set(active) <= lookup.keys():
                raise ValueError(
                    "Candidate contains an orbital outside the measured evidence scope"
                )
            selected[row, [lookup[i] for i in active]] = 1.0
        return selected

    def _coefficients(self, active_list, channel):
        if channel not in self.channels:
            raise ValueError("Unavailable input evidence channel: " + str(channel))
        selected = self._selection(active_list)
        pairs = selected[:, self._first] * selected[:, self._second]
        size = self.context.k
        if channel == "occupation":
            return {"u": selected / size, "b": -2 * pairs / size}
        if channel == "qit":
            entropy_norm, pair_norm = qit_normalizers(size)
            return {
                "h": selected * self._totals["entropy"] / entropy_norm,
                "m": pairs * self._totals["mutual_information"] / pair_norm,
            }
        if channel == "composition":
            return {"composition": selected / size}
        raise ValueError("Unknown evidence response channel: " + str(channel))

    def mean(self, active_list, channel):
        active_list = list(active_list)
        coefficients = self._coefficients(active_list, channel)
        if channel == "composition":
            if (
                self._composition_model != "projection"
                or self.context.composition_preferences is not None
            ):
                return np.asarray(
                    [
                        composition_score(
                            self.context.composition,
                            active,
                            self.context.composition_pair,
                            self._composition_model,
                            composition_channels=getattr(
                                self.context, "composition_channels", None
                            ),
                            composition_preferences=self.context.composition_preferences,
                            composition_purity_weight=self._composition_purity_weight,
                        )
                        for active in active_list
                    ],
                    float,
                )
            return coefficients[channel] @ self._projection[list(self.measured_indices)]
        return sum(value @ self._blocks[name][0] for name, value in coefficients.items())

    def covariance(self, left, right, channel):
        left = self._coefficients(left, channel)
        right = self._coefficients(right, channel)
        result = np.zeros((len(next(iter(left.values()))), len(next(iter(right.values())))))
        if channel == "composition":
            return result
        for name, first in left.items():
            second = right[name]
            _, scale, normalized = self._blocks[name]
            if normalized:
                # Center first, avoiding cancellation in F (I - 11'/d) G'.
                first = first - first.mean(axis=1, keepdims=True)
                second = second - second.mean(axis=1, keepdims=True)
            result += scale * (first @ second.T)
        return result

    def proposal_scores(self, active_list, weights):
        """Exact input descriptors minus the boundary MI, never target labels.

        Used only to rescore the finite pool retained by the QUBO search. This
        cannot recover candidates omitted by its supporting-tangent objective.
        """
        if self._proposal_score_model != "terminal_aligned_v1":
            raise ValueError("Exact proposal rescoring requires terminal-aligned evidence")
        if set(weights) != set(self.channels) or any(
            not _finite_scalar(v) or v <= 0 for v in weights.values()
        ):
            raise ValueError("Proposal rescoring requires one positive weight per channel")
        active_list = list(active_list)
        channels = {c: self.mean(active_list, c) for c in self.channels}
        boundary = np.zeros(len(active_list))
        if "qit" in self.channels:
            selected = self._selection(active_list)
            crossing = np.abs(selected[:, self._first] - selected[:, self._second])
            _, pair_norm = qit_normalizers(self.context.k)
            boundary = (
                crossing @ self._blocks["m"][0] * self._totals["mutual_information"] / pair_norm
            )
        total = sum(weights.values())
        terminal = sum(weights[c] * channels[c] for c in self.channels) / total
        scores = terminal - weights.get("qit", 0.0) * boundary / total
        return dict(
            channels=channels, terminal_score=terminal, cross_boundary_qit=boundary, score=scores
        )


def _condition(observed, prior, prior_variance, noise_variance, normalized=False):
    """For normalized blocks both covariance matrices act in the sum-zero plane."""
    gain = 1 / (1 + noise_variance / prior_variance)
    smaller, larger = sorted((prior_variance, noise_variance))
    variance = smaller / (1 + smaller / larger)
    size = len(observed)
    mean = prior + gain * (observed - prior)
    return dict(
        prior_mean=prior.tolist(),
        observation=observed.tolist(),
        mean=mean.tolist(),
        covariance=dict(
            kind="scaled_sum_zero_projector" if normalized else "scaled_identity",
            dimension=size,
            scale=variance,
            diagonal_variance=variance * (1 - 1 / size) if normalized else variance,
        ),
    )


def statistic_quadratic(context, measured, means, normalization, channel, config):
    """Shared diagnostic/feedback QUBO map; retain the configured QIT scale."""
    measured = list(measured)
    count, size = len(measured), context.k
    first, second = np.triu_indices(count, 1)
    linear, quadratic = np.zeros(context.nmo), np.zeros((context.nmo, context.nmo))
    if channel == "occupation":
        pairs = np.zeros((count, count))
        pairs[first, second] = means["b"]
        pairs[second, first] = means["b"]
        linear[measured] = means["u"] / size
        quadratic[np.ix_(measured, measured)] = -pairs / size
    elif channel == "qit":
        pairs = np.zeros((count, count))
        pairs[first, second] = means["m"]
        pairs[second, first] = means["m"]
        if config.get("proposal_score_model") == "terminal_aligned_v1":
            entropy_norm, pair_norm = qit_normalizers(size)
            hs = normalization["entropy"] / entropy_norm
            ms = normalization["mutual_information"] / pair_norm
            linear[measured] = hs * means["h"] - ms * pairs.sum(axis=1)
            quadratic[np.ix_(measured, measured)] = 1.5 * ms * pairs
        else:
            linear[measured] = 0.5 * (means["h"] - pairs.sum(axis=1))
            quadratic[np.ix_(measured, measured)] = 0.75 * pairs
    else:
        raise ValueError("Unknown uncertain capture-statistic channel")
    return linear, quadratic


def posterior_capture(context, partition, densities, entropies, mutual_information, config):
    """Return filtered capture coefficients and any explicit composition surrogate.

    Ratios are formed before filtering: the latent QIT variables are shares,
    rather than a ratio of separate posterior expectations. u and b already
    include the density-square statistics, so no E[D]**2 substitution occurs.
    This model does not infer, sample, or certify a joint physical density.
    Harmonic composition uses an incumbent-anchored supporting approximation;
    exact nonlinear composition remains the mean used by response inference.
    """
    from .preferences import assert_preference_binding

    assert_preference_binding(context, config)
    channels = _capture_channels(context.backend)
    has_qit = "qit" in channels
    if config["backend"] != context.backend or set(config["channels"]) != set(channels):
        raise ValueError("Capture posterior channels must match the CAS or DMRG physical evidence")
    composition_model = config.get("composition_model", "projection")
    weights = config["weights"]
    if set(weights) != set(channels) or any(not np.isfinite(v) or v <= 0 for v in weights.values()):
        raise ValueError("Capture posterior needs one positive finite weight per channel")
    measured = list(partition["active"])
    count, size, nmo = len(measured), context.k, context.nmo
    if (
        size < 2
        or count < size
        or len(set(measured)) != count
        or any(type(i) is not int for i in measured)
        or not set(measured) <= set(context.window)
    ):
        raise ValueError("Capture posterior requires a complete distinct measured orbital scope")
    electrons = partition.get("electrons")
    if type(electrons) is not int or not 0 <= electrons <= 2 * count:
        raise ValueError("Capture posterior needs its physical diagnostic electron count")
    state_weights = np.asarray(context.states["weights"], float)
    roots = len(state_weights)
    d = validate_densities(densities, state_weights, electrons)
    if d.shape != (roots, count, count):
        raise ValueError(
            "Capture posterior observations must cover every state and measured orbital"
        )
    if has_qit:
        s = np.asarray(entropies, float)
        mi = np.asarray(mutual_information, float)
        if s.shape != (roots, count) or mi.shape != (roots, count, count):
            raise ValueError(
                "Capture posterior observations must cover every state and measured orbital"
            )
        qit(s, mi, state_weights)
        if not np.allclose(mi, mi.swapaxes(1, 2), atol=1e-12, rtol=0) or np.any(
            np.diagonal(mi, axis1=1, axis2=2) != 0
        ):
            raise ValueError("Capture posterior MI must be symmetric with an exact zero diagonal")
    elif entropies is not None or mutual_information is not None:
        raise ValueError("CAS capture cannot supply unsupported QIT evidence")
    projection = np.asarray(context.composition, float)
    if (
        projection.shape != (nmo,)
        or not np.all(np.isfinite(projection))
        or np.any(projection < 0)
        or np.any(projection > 1)
    ):
        raise ValueError("Capture posterior requires complete physical AO projections")
    composition_score(
        context.composition,
        context.active,
        context.composition_pair,
        composition_model,
        composition_channels=getattr(context, "composition_channels", None),
        composition_preferences=context.composition_preferences,
        composition_purity_weight=config.get("composition_purity_weight", 0.5),
    )
    prior_std, noise_std = (config[key] for key in ("evidence_prior_std", "evidence_noise_std"))
    if any(not np.isfinite(v) or v <= 0 for v in (prior_std, noise_std)):
        raise ValueError(
            "Evidence prior and observation standard deviations must be positive and finite"
        )
    try:
        prior_variance, noise_variance = prior_std**2, noise_std**2
    except OverflowError as exc:
        raise ValueError("Evidence variances overflow") from exc
    if any(
        not np.isfinite(v) or v <= 0
        for v in (prior_variance, noise_variance, prior_variance + noise_variance)
    ):
        raise ValueError("Evidence variances must remain finite and nonzero")
    if has_qit:
        entropy = state_weights @ s
        mutual = np.einsum("r,rij->ij", state_weights, mi)
        entropy_scale = float(entropy.sum())
        pair_scale = float(mutual.sum() / 2)
    projection_scale = float(projection[measured].sum())
    scales = (entropy_scale, pair_scale) if has_qit else ()
    if any(not np.isfinite(v) or v <= 0 for v in scales):
        raise ValueError("Capture posterior needs positive measured normalization denominators")
    first, second = np.triu_indices(count, 1)
    pairs = len(first)
    diagonal = np.diagonal(d, axis1=1, axis2=2).real
    observed_u = state_weights @ (2 * diagonal - diagonal**2)
    observed_b = np.einsum("r,rij->ij", state_weights, np.abs(d) ** 2)[first, second]
    nbar = electrons / count
    blocks = {
        "u": _condition(
            observed_u, np.full(count, nbar * (2 - nbar)), prior_variance, noise_variance
        ),
        "b": _condition(observed_b, np.zeros(pairs), prior_variance, noise_variance),
    }
    if has_qit:
        blocks.update(
            {
                "h": _condition(
                    entropy / entropy_scale,
                    np.full(count, 1 / count),
                    prior_variance,
                    noise_variance,
                    normalized=True,
                ),
                "m": _condition(
                    mutual[first, second] / pair_scale,
                    np.full(pairs, 1 / pairs),
                    prior_variance,
                    noise_variance,
                    normalized=True,
                ),
            }
        )
    means = {key: np.asarray(block["mean"]) for key, block in blocks.items()}
    qit_scales = dict(entropy=entropy_scale, mutual_information=pair_scale) if has_qit else {}
    channel_quadratics = {}
    descriptor = None
    for channel in channels:
        linear, quadratic = np.zeros(nmo), np.zeros((nmo, nmo))
        if channel == "composition":
            if composition_model == "projection" and context.composition_preferences is None:
                linear[measured] = projection[measured] / size
            else:
                descriptor, composition = _composition_descriptor(
                    context,
                    composition_model,
                    measured,
                    config.get("composition_purity_weight", 0.5),
                )
                linear[measured] = composition["linear"][measured]
                quadratic[np.ix_(measured, measured)] = composition["pair"][
                    np.ix_(measured, measured)
                ]
                if composition["constant"] != 0.0:
                    raise ValueError("Capture composition requires a homogeneous supporting plane")
        else:
            linear, quadratic = statistic_quadratic(
                context, measured, means, qit_scales, channel, config
            )
        channel_quadratics[channel] = dict(
            constant=0.0, linear=linear.tolist(), quadratic=quadratic.tolist()
        )
    total_weight = sum(weights.values())
    linear = (
        sum(weights[c] * np.asarray(channel_quadratics[c]["linear"]) for c in channels)
        / total_weight
    )
    quadratic = (
        sum(weights[c] * np.asarray(channel_quadratics[c]["quadratic"]) for c in channels)
        / total_weight
    )
    binding = dict(
        context_id=context.context_id,
        basis_id=context.basis_id,
        geometry_id=context.geometry_id,
        states=context.states,
        measured_indices=measured,
        electrons=electrons,
    )
    raw_hash = identity(
        dict(
            **binding,
            densities_real=d.real.tolist(),
            densities_imag=d.imag.tolist(),
            entropies=s.tolist() if has_qit else None,
            mutual_information=mi.tolist() if has_qit else None,
            projection=projection.tolist(),
        )
    )
    protocol_keys = (
        ("backend", "dmrg_probe_m", "dmrg_max_sweeps", "thr_energy", "thr_ci")
        if has_qit
        else ("backend", "thr_energy", "thr_ci")
    )
    declared_protocol = {key: config.get(key) for key in protocol_keys}
    model = dict(
        prior_std=prior_std,
        noise_std=noise_std,
        prior="exchangeable_uniform_density_and_normalized_QIT_shares"
        if has_qit
        else "exchangeable_uniform_density",
        approximation="Gaussian_capture_statistics",
    )
    audit = dict(
        schema="pas_capture_statistics_posterior_v1",
        **binding,
        raw_evidence_hash=raw_hash,
        declared_protocol_hash=identity(declared_protocol),
        model_hash=identity(model),
        **model,
        gain=1 / (1 + noise_variance / prior_variance),
        blocks=blocks,
        pair_order="upper_triangle_row_major_in_measured_order",
        normalized_blocks=["h", "m"] if has_qit else [],
        observation_count=1,
        replay_semantics="reconstruct_once_from_frozen_prior_not_incremental_update",
        calibrated=False,
        finite_M_error_certified=False,
        physical_matrix_posterior=False,
        gaussian_tails_physically_constrained=False,
        cross_block_covariance="assumed_zero",
        output_quality_observation=False,
        composition_exact=True,
        normalization="condition_on_normalized_statistics_not_ratio_of_posterior_means",
    )
    normalization = (
        dict(
            entropy=entropy_scale,
            mutual_information=pair_scale,
            projection=projection_scale,
            aggregation="state_weighted_then_share_normalized_then_conditioned",
        )
        if has_qit
        else dict(
            projection=projection_scale,
            aggregation="state_weighted_density_statistics_then_conditioned",
        )
    )
    result = dict(
        constant=0.0,
        linear=linear.tolist(),
        quadratic=quadratic.tolist(),
        quadratic_convention="xQx",
        scope="input_capture_statistics_posterior_not_target_CAS_truth",
        score_model="posterior_evidence_capture_v1",
        measured_indices=measured,
        candidate_scope="must_be_subset_of_measured_indices",
        score_channels=list(channels),
        evidence_channels=list(channels),
        cut_penalty=1.0,
        normalization=normalization,
        composition_scale="terminal_score",
        score_definition="E[weighted_mean(Un,P_A/K,0.5*(h_A+m_internal-m_cut))|one_survey]",
        occupation_statistic="posterior_of_2Dii_minus_Dii_squared_and_abs_Dij_squared_not_square_of_mean_density",
        mutual_information_convention="half_mutual_information_natural_log",
        channel_quadratics=channel_quadratics,
        input_posterior=audit,
    )
    if descriptor is not None:
        result["composition_descriptor"] = descriptor
        result["score_definition"] = (
            "E[weighted_mean(Un,terminal_composition_surrogate,0.5*(h_A+m_internal-m_cut))|one_survey]"
        )
    if not has_qit:
        result.pop("cut_penalty")
        result.pop("mutual_information_convention")
        result["score_definition"] = (
            "E[weighted_mean(Un,terminal_composition_surrogate)|one_survey]"
            if descriptor is not None
            else "E[weighted_mean(Un,P_A/K)|one_survey]"
        )
        audit.pop("finite_M_error_certified")
        audit["normalization"] = "condition_on_density_statistics"
    if config.get("proposal_score_model") == "terminal_aligned_v1":
        result["proposal_score_model"] = config["proposal_score_model"]
        result["candidate_rescoring"] = "exact_terminal_descriptors_minus_cross_boundary_MI"
        result["score_definition"] = (
            "weighted_mean(Un,terminal_composition_surrogate,QIT_terminal_minus_boundary)"
        )
        if has_qit:
            result["qit_scale"] = "terminal_K_pair_count_ln4"
        else:
            result["score_definition"] = "weighted_mean(Un,terminal_composition_surrogate)"
    return result
