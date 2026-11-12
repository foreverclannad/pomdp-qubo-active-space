"""Exact latent-input marginals of the existing diagnostic + residual GP.

These are derived views of one joint conditioning operation, not new survey
observations. Never bind them back as independent priors or discard the
posterior input/residual cross covariance when predicting total quality.
"""

from copy import deepcopy

import numpy as np

from .common import identity


class InputPosterior:
    """Compact block posterior: diagnostic covariance minus low-rank factors.

    Factors with the same channel share training columns, so their products
    retain induced cross-block covariance (u/b and h/m). Channels stay independent
    under the existing model. This is not a physically constrained density.
    """

    def __init__(self, belief, context_id=None):
        cid = context_id or belief.context.context_id
        evidence = belief._input(cid)
        self._fields_id = belief._input_evidence_ids[cid]
        self._evidence = deepcopy(evidence)
        self.means, self.factors, self.channels = {}, {}, {}
        self.training = {}
        for channel in belief.config["channels"]:
            if channel == "composition":
                continue  # The AO input composition is deterministic, not measured noise.
            active, _, _, factor, alpha, _ = belief._system(channel)
            rows = belief.observations[channel]
            self.training[channel] = deepcopy(rows)
            indices = [i for i, row in enumerate(rows) if row["context_id"] == cid]
            coefficients = evidence._coefficients([active[i] for i in indices], channel)
            for name, local in coefficients.items():
                prior, scale, normalized = evidence._blocks[name]
                cross = np.zeros((len(prior), len(rows)))
                if normalized:
                    local = local - local.mean(axis=1, keepdims=True)
                cross[:, indices] = scale * local.T
                self.means[name] = prior + cross @ alpha if rows else prior.copy()
                self.factors[name] = (
                    np.linalg.solve(factor, cross.T).T if rows else np.empty((len(prior), 0))
                )
                self.channels[name] = channel

    def covariance(self, left, right=None):
        """Materialize only the requested block; compact audit need not be dense."""
        right = left if right is None else right
        first, second = self.means[left], self.means[right]
        value = np.zeros((len(first), len(second)))
        if left == right:
            _, scale, normalized = self._evidence._blocks[left]
            value = scale * (np.eye(len(first)) - (1 / len(first) if normalized else 0))
        if self.channels[left] == self.channels[right]:
            value -= self.factors[left] @ self.factors[right].T
        return value

    def audit(self):
        blocks = {}
        for name, mean in self.means.items():
            prior, scale, normalized = self._evidence._blocks[name]
            factor = self.factors[name]
            prior_diagonal = scale * (1 - 1 / len(mean) if normalized else 1)
            diagonal = prior_diagonal - np.einsum("ij,ij->i", factor, factor)
            if np.min(diagonal, initial=0) < -1e-9:
                raise RuntimeError("Negative input posterior variance")
            blocks[name] = dict(
                mean=mean.tolist(),
                mean_shift=(mean - prior).tolist(),
                prior_covariance=dict(
                    kind="scaled_sum_zero_projector" if normalized else "scaled_identity",
                    scale=scale,
                    dimension=len(mean),
                ),
                downdate_factor=factor.tolist(),
                diagonal_variance=diagonal.tolist(),
                channel=self.channels[name],
            )
        return dict(
            schema="pas_joint_input_posterior_v1",
            context_id=self._evidence.context_id,
            evidence_id=self._evidence.evidence_id,
            diagnostic_fields_id=self._fields_id,
            blocks=blocks,
            training=self.training,
            covariance_convention="block_diagonal_diagnostic_prior_minus_F_left_F_right_transpose_within_channel",
            cross_block_covariance="retained_within_channel",
            independent_channels=True,
            normalized_blocks=[name for name, block in self._evidence._blocks.items() if block[2]],
            composition_exact=True,
            calibrated=False,
            physical_matrix_posterior=False,
            gaussian_tails_physically_constrained=False,
            not_for_rebinding_as_prior=True,
            assimilation="derived_once_from_frozen_diagnostics_and_unique_response_receipts",
        )

    def proposal_scores(self, active, weights):
        # Only use the descriptor's mean-based scoring method. Its covariance
        # remains diagnostic, whereas this object's covariance() is conditioned.
        descriptor = deepcopy(self._evidence)
        descriptor._blocks = {
            name: (self.means[name], scale, normalized)
            for name, (_, scale, normalized) in descriptor._blocks.items()
        }
        return descriptor.proposal_scores(active, weights)

    def proposal_fields(self, fields, config):
        """Rebuild only uncertain input terms, preserving composition and scales."""
        from .evidence_belief import statistic_quadratic

        if identity(fields) != self._fields_id:
            raise ValueError("Feedback proposal evidence differs from the bound diagnostic")
        result = deepcopy(fields)
        changed = any(
            not np.array_equal(self.means[name], block[0])
            for name, block in self._evidence._blocks.items()
        )
        if changed:
            for channel in config["channels"]:
                if channel != "composition":
                    linear, pair = statistic_quadratic(
                        self._evidence.context,
                        self._evidence.measured_indices,
                        self.means,
                        fields["normalization"],
                        channel,
                        config,
                    )
                    result["channel_quadratics"][channel] = dict(
                        constant=0.0, linear=linear.tolist(), quadratic=pair.tolist()
                    )
            quads, weights = result["channel_quadratics"], config["weights"]
            total = sum(weights.values())
            for key in ("linear", "quadratic"):
                result[key] = (
                    sum(weights[c] * np.asarray(quads[c][key]) for c in config["channels"]) / total
                ).tolist()
        result["joint_input_posterior"] = self.audit()
        result["joint_input_posterior"]["proposal_coefficients_changed"] = changed
        result["score_definition"] += "|joint_target_response_conditioning"
        return result


def joint_prediction(belief, active_list, kinds, channel):
    """Return input, residual and their cross covariance, plus unchanged totals."""
    total_mean, total_cov = belief.predict_mixed(active_list, kinds, channel)
    active = [belief._active(a) for a in active_list]
    cid = belief.context.context_id
    evidence = belief._input(cid)
    training, train_kinds, _, factor, alpha, _ = belief._system(channel)
    input_mean = evidence.mean(active, channel)
    input_cov = evidence.covariance(active, active, channel)
    residual_mean = np.zeros(len(active))
    residual_cov = belief._prior_cov(active, kinds, active, kinds, channel) - input_cov
    cross_cov = np.zeros_like(input_cov)
    if training:
        contexts = [row["context_id"] for row in belief.observations[channel]]
        input_cross = belief._input_covariance(
            active, training, channel, [cid] * len(active), contexts
        )
        residual_cross = (
            belief._prior_cov(
                active, kinds, training, train_kinds, channel, right_contexts=contexts
            )
            - input_cross
        )
        input_mean = input_mean + input_cross @ alpha
        residual_mean = residual_cross @ alpha
        left = np.linalg.solve(factor, input_cross.T)
        right = np.linalg.solve(factor, residual_cross.T)
        input_cov = input_cov - left.T @ left
        residual_cov = residual_cov - right.T @ right
        cross_cov = -left.T @ right
    if not np.allclose(
        input_mean + residual_mean, total_mean, atol=1e-10, rtol=1e-10
    ) or not np.allclose(
        input_cov + residual_cov + cross_cov + cross_cov.T, total_cov, atol=1e-10, rtol=1e-10
    ):
        raise RuntimeError(
            "Joint input/residual decomposition differs from the original total posterior"
        )
    return dict(
        input_mean=input_mean,
        input_covariance=input_cov,
        residual_mean=residual_mean,
        residual_covariance=residual_cov,
        input_residual_covariance=cross_cov,
        total_mean=total_mean,
        total_covariance=total_cov,
    )
