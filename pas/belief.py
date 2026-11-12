"""Gaussian response beliefs with optional authenticated cross-context sharing.

The polynomial prior is an uncalibrated response model, not a model of
measurement noise. Composition at the input is a known deterministic value.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .common import identity
from .config import configuration
from .metrics import (
    composition_components,
    composition_proposal_weights,
    composition_quadratic,
    composition_score,
)


@dataclass
class Quadratic:
    """Value = constant + linear @ x + x @ pair @ x; pair has zero diagonal."""

    constant: float
    linear: np.ndarray
    pair: np.ndarray

    def __call__(self, active):
        a = np.asarray(tuple(active), dtype=int)
        return float(self.constant + self.linear[a].sum() + self.pair[np.ix_(a, a)].sum())

    def scaled(self, weight):
        return Quadratic(weight * self.constant, weight * self.linear, weight * self.pair)

    def __add__(self, other):
        return Quadratic(
            self.constant + other.constant, self.linear + other.linear, self.pair + other.pair
        )


def kernel(left, right, k):
    """Unit-amplitude squared projector kernel in the fixed orthonormal MO basis."""
    return np.asarray(
        [[(1.0 + len(set(a) & set(b)) / k) ** 2 / 4.0 for b in right] for a in left], dtype=float
    ).reshape(len(left), len(right))


class ResponseBelief:
    def __init__(self, context, config):
        self.context = context = deepcopy(context)
        self.config = configuration(config)
        from .preferences import assert_preference_binding

        assert_preference_binding(context, self.config)
        self._shared = (
            self.config.get("response_model", "context_local_v1") == "shared_projectors_v1"
        )
        self._contexts = {context.context_id: context}
        self._frames = {}
        self._losses = {}
        if self._shared and set(self.config["channels"]) & {"occupation", "composition"}:
            from .projector_response import ProjectorFrame

            self._frames[context.context_id] = ProjectorFrame(context)
        if "density_response_model" in self.config:
            from .density_response import DensityLoss

            self._losses[context.context_id] = DensityLoss(context)
        if "composition" in self.config["channels"]:
            if context.composition is None or not np.isfinite(context.composition).all():
                raise ValueError("Enabled composition needs known input AO projections")
            self._active(context.active)
            self.composition(context.active)
        if min(self.config["prior_std"], self.config["discrepancy_std"]) <= 0:
            raise ValueError("Prior standard deviations must be positive")
        if not 0 < self.config["jitter"] <= self.config["max_jitter"]:
            raise ValueError("Invalid numerical jitter interval")
        self.observations = {c: [] for c in self.config["channels"]}
        # Different computational scopes have different success rates. The
        # whole-window survey never calibrates target-CAS probe/optimize success.
        self.validity = {k: [1, 1] for k in ("probe", "optimize", "survey")}
        self.evidence_ids = {}
        self.validity_evidence = {}
        self.request_evidence = {}
        self._points = {}
        self._cache = {}
        self.jitter_used = {}
        self._input_evidence = {}
        self._input_evidence_ids = {}

    def _register_context(self, source):
        from .preferences import assert_preference_binding

        assert_preference_binding(source, self.config)
        source_id = source.context_id
        if not self._shared and source_id != self.context.context_id:
            raise ValueError("Quality observation belongs to another orbital context")
        if source_id in self._contexts:
            if identity(source.to_dict()) != identity(self._contexts[source_id].to_dict()):
                raise ValueError("Conflicting original context payload")
            return
        if self._frames:
            from .projector_response import ProjectorFrame

            frame = ProjectorFrame(source)
            self._frames[self.context.context_id].assert_compatible(
                frame, composition="composition" in self.config["channels"]
            )
            self._frames[source_id] = frame
        elif any(
            getattr(source, field) != getattr(self.context, field)
            for field in ("backend", "geometry_id", "protocol_id", "effective_config_id", "states")
        ):
            raise ValueError("Incompatible response contexts")
        self._contexts[source_id] = deepcopy(source)
        if self._losses:
            from .density_response import DensityLoss

            self._losses[source_id] = DensityLoss(source)

    def bind_input_evidence(self, context, fields):
        """Freeze a survey-derived response prior, without creating a target label."""
        if "input_response_model" not in self.config:
            raise ValueError("Input response is disabled")
        if "joint_input_posterior" in fields:
            raise ValueError(
                "A joint input posterior cannot be rebound as an independent diagnostic prior"
            )
        descriptor = fields.get("composition_descriptor")
        if descriptor is not None and not isinstance(descriptor, dict):
            raise ValueError("Malformed input composition descriptor")
        if (descriptor or {}).get("model", "projection") != self.config["composition_model"]:
            raise ValueError("Input response composition differs from the frozen scoring model")
        from .evidence_belief import EvidenceResponse, validate_proposal_score_model

        validate_proposal_score_model(fields, self.config)
        evidence = EvidenceResponse(context, fields)
        if list(evidence.channels) != self.config["channels"]:
            raise ValueError("Input evidence channels differ from the frozen response channels")
        self._register_context(context)
        cid, digest = context.context_id, identity(fields)
        if cid in self._input_evidence:
            if self._input_evidence_ids[cid] != digest:
                raise ValueError("Conflicting frozen input-response evidence")
            return False
        self._input_evidence[cid] = evidence
        self._input_evidence_ids[cid] = digest
        self._cache.clear()
        return True

    def _input(self, context_id):
        if context_id not in self._input_evidence:
            raise ValueError("Input response requires bound evidence in the original input context")
        return self._input_evidence[context_id]

    def _input_covariance(self, left, right, channel, left_contexts, right_contexts):
        result = np.zeros((len(left), len(right)))
        # Each survey's errors are shared by its candidates, not by different
        # bases. Cross-basis transfer remains in the existing residual kernel.
        for cid in set(left_contexts) | set(right_contexts):
            self._input(cid)
        for cid in set(left_contexts) & set(right_contexts):
            il = [i for i, c in enumerate(left_contexts) if c == cid]
            ir = [i for i, c in enumerate(right_contexts) if c == cid]
            result[np.ix_(il, ir)] = self._input(cid).covariance(
                [left[i] for i in il], [right[i] for i in ir], channel
            )
        return result

    def _active(self, active):
        return tuple(self.context.partition(active)["active"])

    def composition(self, active, context=None):
        context = self.context if context is None else context
        return composition_score(
            context.composition,
            active,
            context.composition_pair,
            self.config["composition_model"],
            composition_channels=getattr(context, "composition_channels", None),
            composition_preferences=context.composition_preferences,
            composition_purity_weight=self.config.get("composition_purity_weight", 0.5),
        )

    def _loss_feature(self, active, contexts=None, kinds=None, channel=None):
        contexts = contexts if contexts is not None else [self.context.context_id] * len(active)
        value = np.asarray(
            [
                self._losses[c].values([a])[0] / self._losses[c].scale
                for a, c in zip(active, contexts)
            ]
        )
        if channel == "composition" and kinds is not None:
            value *= np.asarray([kind == "optimize" for kind in kinds])
        return value

    def _loss_quadratic(self, scale):
        loss = self._losses[self.context.context_id]
        affine = scale * loss.affine() / loss.scale
        return Quadratic(
            float(affine[0]), affine[1:], np.zeros((self.context.nmo, self.context.nmo))
        )

    def _mean(self, active, channel, contexts=None, kinds=None):
        if "input_response_model" in self.config:
            contexts = contexts if contexts is not None else [self.context.context_id] * len(active)
            return np.asarray(
                [self._input(cid).mean([a], channel)[0] for a, cid in zip(active, contexts)],
                dtype=float,
            )
        if channel == "composition":
            contexts = contexts if contexts is not None else [self.context.context_id] * len(active)
            value = np.asarray(
                [self.composition(a, self._contexts[c]) for a, c in zip(active, contexts)]
            )
        else:
            value = np.full(len(active), float(self.config["prior_mean"]))
        if self._losses:
            value += self.config["density_slope_mean"] * self._loss_feature(
                active, contexts, kinds, channel
            )
        return value

    def _features(self, active, contexts):
        dimension = self._frames[self.context.context_id].affine_features().shape[1]
        return np.asarray([self._frames[c].feature(a) for a, c in zip(active, contexts)]).reshape(
            len(active), dimension
        )

    def _prior_cov(
        self, left, left_kinds, right, right_kinds, channel, left_contexts=None, right_contexts=None
    ):
        left_contexts = (
            left_contexts if left_contexts is not None else [self.context.context_id] * len(left)
        )
        right_contexts = (
            right_contexts if right_contexts is not None else [self.context.context_id] * len(right)
        )
        input_cov = (
            self._input_covariance(left, right, channel, left_contexts, right_contexts)
            if "input_response_model" in self.config
            else 0.0
        )
        value = self.config["prior_std"] ** 2 * np.ones((len(left), len(right)))
        lp = np.asarray([k == "probe" for k in left_kinds])
        rp = np.asarray([k == "probe" for k in right_kinds])
        if self._shared and channel != "qit":
            left_contexts = (
                left_contexts
                if left_contexts is not None
                else [self.context.context_id] * len(left)
            )
            right_contexts = (
                right_contexts
                if right_contexts is not None
                else [self.context.context_id] * len(right)
            )
            same = np.equal.outer(left_contexts, right_contexts)
            base = (
                self._features(left, left_contexts) @ self._features(right, right_contexts).T
            ) ** 2
            # Half the variance is shared; half represents context-specific
            # optimizer/gauge response. Fidelity discrepancy is context-local.
            value *= 0.5 + 0.5 * same
            if channel == "composition":
                value *= np.outer(~lp, ~rp)
            else:
                value += self.config["discrepancy_std"] ** 2 * same * np.outer(lp, rp)
            value *= base
            if self._losses:
                value += self.config["density_slope_std"] ** 2 * np.outer(
                    self._loss_feature(left, left_contexts, left_kinds, channel),
                    self._loss_feature(right, right_contexts, right_kinds, channel),
                )
            return value + input_cov
        if channel == "composition":
            value *= np.outer(~lp, ~rp)
        else:
            value += self.config["discrepancy_std"] ** 2 * np.outer(lp, rp)
        return value * kernel(left, right, self.context.k) + input_cov

    @staticmethod
    def _receipt_identity(receipt):
        if not receipt.get("evidence_id") or not receipt.get("request_id"):
            raise ValueError("A physical receipt requires evidence_id and request_id")
        return str(receipt["evidence_id"]), identity(receipt)

    def observe_validity(self, receipt):
        """Replay attempt validity, including old contexts; never imports quality."""
        kind = receipt.get("kind")
        if kind not in self.validity:
            raise ValueError(f"Unknown action kind: {kind}")
        eid, digest = self._receipt_identity(receipt)
        request = str(receipt["request_id"])
        if request in self.request_evidence and self.request_evidence[request] != eid:
            raise ValueError("One physical request cannot create multiple evidence identities")
        if eid in self.validity_evidence:
            if self.validity_evidence[eid] != digest:
                raise ValueError("Conflicting evidence identity")
            return False
        valid = bool(receipt.get("valid"))
        if valid and not (receipt.get("converged") and receipt.get("state_valid")):
            raise ValueError("Valid receipt must certify convergence and state identity")
        self.validity[kind][0 if valid else 1] += 1
        self.validity_evidence[eid] = digest
        self.request_evidence[request] = eid
        return True

    def observe(self, receipt, input_context=None):
        """Condition on the original request input, never relabel output orbitals."""
        source = self.context if input_context is None else deepcopy(input_context)
        source_id = source.context_id
        if receipt.get("context_id") != source_id:
            raise ValueError("Quality observation belongs to another orbital context")
        if not self._shared and source_id != self.context.context_id:
            raise ValueError("Quality observation belongs to another orbital context")
        if source_id in self._contexts and identity(source.to_dict()) != identity(
            self._contexts[source_id].to_dict()
        ):
            raise ValueError("Conflicting original context payload")
        if receipt.get("kind") not in ("probe", "optimize"):
            raise ValueError("Only target-CAS probe/optimize receipts update this belief")
        eid, digest = self._receipt_identity(receipt)
        if eid in self.evidence_ids:
            if self.evidence_ids[eid] != digest:
                raise ValueError("Conflicting evidence identity")
            return False
        self._register_context(source)
        if "input_response_model" in self.config:
            self._input(source_id)
        active = tuple(source.partition(receipt["active"])["active"])
        kind = receipt["kind"]
        updates = []
        if receipt.get("valid"):
            if not (receipt.get("converged") and receipt.get("state_valid")):
                raise ValueError("Valid receipt lacks physical validity checks")
            metrics = receipt.get("metrics", {})
            for channel in self.observations:
                if channel not in metrics:
                    raise ValueError(f"Missing enabled {channel} quality")
                value = float(metrics[channel])
                if not np.isfinite(value) or not -1e-10 <= value <= 1 + 1e-10:
                    raise ValueError(f"Invalid observed {channel} quality")
                if channel == "qit" and source_id != self.context.context_id:
                    continue
                if channel == "composition" and kind == "probe":
                    if abs(value - self.composition(active, source)) > 1e-7:
                        raise ValueError("Fixed-orbital composition disagrees with the input")
                    continue
                key = (source_id, kind, active, channel)
                if key in self._points:
                    if abs(value - self._points[key]) > 1e-10:
                        raise ValueError("Conflicting deterministic response at the same point")
                    continue
                updates.append((key, channel, value))
        # Check identity before mutating either the quality or validity posterior.
        if eid in self.validity_evidence and self.validity_evidence[eid] != digest:
            raise ValueError("Conflicting evidence identity")
        self.observe_validity(receipt)
        for key, channel, value in updates:
            self._points[key] = value
            self.observations[channel].append(
                dict(active=active, kind=kind, value=value, evidence_id=eid, context_id=source_id)
            )
            self._cache.pop(channel, None)
        self.evidence_ids[eid] = digest
        return True

    def success_probability(self, kind):
        a, b = self.validity[kind]
        return a / (a + b)

    def _system(self, channel):
        if channel not in self.observations:
            raise ValueError(f"Disabled channel: {channel}")
        if channel in self._cache:
            return self._cache[channel]
        rows = self.observations[channel]
        active, kinds = [r["active"] for r in rows], [r["kind"] for r in rows]
        contexts = [r["context_id"] for r in rows]
        y = np.asarray([r["value"] for r in rows])
        if not rows:
            result = (active, kinds, y, None, np.empty(0), 0.0)
        else:
            cov = self._prior_cov(active, kinds, active, kinds, channel, contexts, contexts)
            jitter = self.config["jitter"]
            while True:
                try:
                    factor = np.linalg.cholesky(cov + jitter * np.eye(len(rows)))
                    break
                except np.linalg.LinAlgError as exc:
                    if jitter * 10 > self.config["max_jitter"] * (1 + 1e-12):
                        raise RuntimeError(
                            "Gaussian conditioning failed at maximum numerical jitter"
                        ) from exc
                    jitter *= 10
            centered = y - self._mean(active, channel, contexts, kinds)
            alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, centered))
            result = (active, kinds, y, factor, alpha, jitter)
        self.jitter_used[channel] = result[-1]
        self._cache[channel] = result
        return result

    def predict_mixed(self, active_list, kinds, channel):
        active = [self._active(a) for a in active_list]
        if len(active) != len(kinds) or any(k not in ("probe", "optimize") for k in kinds):
            raise ValueError("One valid action kind is required per query")
        training, train_kinds, _, factor, alpha, _ = self._system(channel)
        mean = self._mean(active, channel, kinds=kinds)
        covariance = self._prior_cov(active, kinds, active, kinds, channel)
        if training:
            cross = self._prior_cov(
                active,
                kinds,
                training,
                train_kinds,
                channel,
                right_contexts=[r["context_id"] for r in self.observations[channel]],
            )
            mean = mean + cross @ alpha
            solved = np.linalg.solve(factor, cross.T)
            covariance -= solved.T @ solved
        covariance = (covariance + covariance.T) / 2
        if len(active) and np.min(np.diag(covariance)) < -1e-9:
            raise RuntimeError("Negative Gaussian posterior variance")
        np.fill_diagonal(covariance, np.maximum(np.diag(covariance), 0))
        return mean, covariance

    def predict(self, active_list, kind, channel):
        return self.predict_mixed(active_list, [kind] * len(active_list), channel)

    def _kernel_quadratic(self, active, scale):
        b = np.zeros(self.context.nmo)
        b[list(active)] = 1 / (2 * self.context.k)
        pair = scale * np.outer(b, b)
        linear = scale * b + np.diag(pair)
        np.fill_diagonal(pair, 0)
        return Quadratic(scale / 4, linear, pair)

    def _mean_quadratic(self, channel):
        """QUBO proposal mean; harmonic composition uses an anchored upper plane."""
        linear = np.zeros(self.context.nmo)
        pair = np.zeros((self.context.nmo, self.context.nmo))
        constant = self.config["prior_mean"]
        if channel == "composition":
            constant = 0.0
            if (
                self.config["composition_model"] == "harmonic_channel_pair_v1"
                or self.context.composition_preferences is not None
            ):
                values = composition_quadratic(
                    self.context.composition,
                    self.context.active,
                    self.context.composition_pair,
                    self.config["composition_model"],
                    composition_channels=self.context.composition_channels,
                    composition_preferences=self.context.composition_preferences,
                    composition_purity_weight=self.config.get("composition_purity_weight", 0.5),
                )
                constant, linear, pair = values["constant"], values["linear"], values["pair"]
            else:
                a, b = composition_proposal_weights(
                    self.context.composition,
                    self.context.active,
                    self.context.composition_pair,
                    self.config["composition_model"],
                    composition_purity_weight=self.config.get("composition_purity_weight", 0.5),
                )
                linear = a * (np.asarray(self.context.composition, dtype=float) / self.context.k)
                if self.context.composition_pair is not None:
                    pair = b * np.asarray(self.context.composition_pair, dtype=float)
        result = Quadratic(float(constant), linear, pair)
        if self._losses:
            result = result + self._loss_quadratic(self.config["density_slope_mean"])
        return result

    def composition_proposal_audit(self):
        """Bind the local approximation to this accepted context, not a candidate."""
        if self.config["composition_model"] not in (
            "harmonic_determinant_pair_v1",
            "harmonic_channel_pair_v1",
        ):
            raise ValueError("Harmonic proposal audit requires the harmonic composition model")
        p, b = composition_components(
            self.context.composition,
            self.context.active,
            self.context.composition_pair,
            self.config["composition_model"],
            composition_channels=getattr(self.context, "composition_channels", None),
        )
        weights = composition_proposal_weights(
            self.context.composition,
            self.context.active,
            self.context.composition_pair,
            self.config["composition_model"],
            composition_channels=getattr(self.context, "composition_channels", None),
            composition_purity_weight=self.config.get("composition_purity_weight", 0.5),
        )
        result = dict(
            schema="pas_harmonic_composition_proposal_v1",
            exact_objective=False,
            anchor_context_id=self.context.context_id,
            anchor_active=list(self.context.active),
            anchor_projection=p,
            anchor_pair=b,
            projection_weight=weights[0],
            pair_weight=weights[1],
            approximation="tangent_upper_plane" if p + b > 0.0 else "symmetric_support_at_origin",
            coefficients_normalized=False,
            exact_mean_used_for_planning_and_learning=True,
        )
        if self.config["composition_model"] == "harmonic_channel_pair_v1":
            result.update(
                model=self.config["composition_model"],
                approximation="nested_supporting_tangents",
                channels_id=identity(self.context.composition_channels),
            )
        if self.context.composition_preferences is not None:
            result["composition_preferences_id"] = identity(self.context.composition_preferences)
        return result

    def _prior_draw(self, std, rng):
        # In this basis z(x)=[1/sqrt(2), x/sqrt(2K)].
        n = self.context.nmo
        scale = np.r_[1 / np.sqrt(2), np.full(n, 1 / np.sqrt(2 * self.context.k))]
        matrix = rng.normal(size=(n + 1, n + 1)) * std
        matrix = (matrix + matrix.T) / 2 * np.outer(scale, scale)
        pair = matrix[1:, 1:].copy()
        linear = 2 * matrix[0, 1:] + np.diag(pair)
        np.fill_diagonal(pair, 0)
        return Quadratic(float(matrix[0, 0]), linear, pair)

    @staticmethod
    def _dense_quadratic(matrix):
        pair = matrix[1:, 1:].copy()
        linear = matrix[0, 1:] + matrix[1:, 0] + np.diag(pair)
        np.fill_diagonal(pair, 0)
        return Quadratic(float(matrix[0, 0]), linear, pair)

    def _shared_draw(self, channel, rng):
        """One joint prior draw at current coefficients and original history."""
        from .projector_response import polynomial_draw

        rows = self.observations[channel]
        contexts = [r["context_id"] for r in rows]
        features = self._features([r["active"] for r in rows], contexts)
        affine = self._frames[self.context.context_id].affine_features()
        matrix, history = polynomial_draw(
            affine, features, self.config["prior_std"] / np.sqrt(2), rng
        )
        # Independent residuals for each original input context. The current
        # context also gets its residual when it has no previous labels.
        for cid in sorted(set(contexts) | {self.context.context_id}):
            selected = np.asarray([i for i, c in enumerate(contexts) if c == cid], dtype=int)
            query = affine if cid == self.context.context_id else affine[:0]
            local, values = polynomial_draw(
                query, features[selected], self.config["prior_std"] / np.sqrt(2), rng
            )
            history[selected] += values
            if cid == self.context.context_id:
                matrix += local
            if channel != "composition":
                low = np.asarray([i for i in selected if rows[i]["kind"] == "probe"], dtype=int)
                if len(low):
                    _, values = polynomial_draw(
                        affine[:0], features[low], self.config["discrepancy_std"], rng
                    )
                    history[low] += values
        result = self._dense_quadratic(matrix)
        if self._losses:
            slope = float(rng.normal(scale=self.config["density_slope_std"]))
            result = result + self._loss_quadratic(slope)
            history += slope * self._loss_feature(
                [r["active"] for r in rows], contexts, [r["kind"] for r in rows], channel
            )
        return result, history

    def posterior_quadratic(self, channel, rng=None):
        """Proposal mean or draw: exact Gaussian residual, possibly approximate mean.

        Only harmonic composition replaces its nonlinear known mean with the
        current accepted-context tangent. Conditioning always subtracts the
        exact original-context mean. predict() retains the exact harmonic mean.
        """
        if "input_response_model" in self.config:
            raise ValueError(
                "Input response uses joint predict(); QUBO proposals use posterior capture"
            )
        active, kinds, y, factor, alpha, jitter = self._system(channel)
        result = self._mean_quadratic(channel)
        if self._shared and channel != "qit":
            contexts = [r["context_id"] for r in self.observations[channel]]
            if rng is not None:
                draw, prior = self._shared_draw(channel, rng)
                result = result + draw
                if active:
                    residual = y - self._mean(active, channel, contexts, kinds) - prior
                    residual -= rng.normal(scale=np.sqrt(jitter), size=len(active))
                    alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, residual))
            affine = self._frames[self.context.context_id].affine_features()
            for a, cid, coefficient in zip(active, contexts, alpha):
                v = affine @ self._frames[cid].feature(a)
                scale = (
                    coefficient
                    * self.config["prior_std"] ** 2
                    * (0.5 + 0.5 * (cid == self.context.context_id))
                )
                result = result + self._dense_quadratic(scale * np.outer(v, v))
            if self._losses and active:
                scale = self.config["density_slope_std"] ** 2 * float(
                    self._loss_feature(active, contexts, kinds, channel) @ alpha
                )
                result = result + self._loss_quadratic(scale)
            return result
        if rng is not None:
            shared = self._prior_draw(self.config["prior_std"], rng)
            discrepancy = (
                self._prior_draw(self.config["discrepancy_std"], rng)
                if channel != "composition"
                else None
            )
            result = result + shared
            if active:
                prior = self._mean(active, channel) + np.asarray(
                    [
                        shared(a)
                        + (discrepancy(a) if kind == "probe" and discrepancy is not None else 0)
                        for a, kind in zip(active, kinds)
                    ]
                )
                # Auxiliary jitter reproduces the numerically regularized covariance;
                # it is not physical observation noise or new experimental evidence.
                residual = y - prior - rng.normal(scale=np.sqrt(jitter), size=len(active))
                alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, residual))
        for a, coefficient in zip(active, alpha):
            result = result + self._kernel_quadratic(a, coefficient * self.config["prior_std"] ** 2)
        return result

    def input_posterior(self, context_id=None):
        """Condition latent input statistics on the existing unique target receipts."""
        from .input_feedback import InputPosterior

        return InputPosterior(self, context_id)

    def predict_input_response(self, active_list, kinds, channel):
        from .input_feedback import joint_prediction

        return joint_prediction(self, active_list, kinds, channel)

    def input_response_audit(self, pool):
        """Expose the exact prior and conditioned moments used by the planner."""
        evidence = self._input(self.context.context_id)
        channels = {}
        for channel in self.config["channels"]:
            mean, covariance = self.predict(pool, "optimize", channel)
            channels[channel] = dict(
                input_mean=evidence.mean(pool, channel).tolist(),
                input_covariance=evidence.covariance(pool, pool, channel).tolist(),
                response_mean=mean.tolist(),
                response_covariance=covariance.tolist(),
                response_observation_count=len(self.observations[channel]),
            )
        result = dict(
            model=self.config["input_response_model"],
            calibrated=False,
            input_is_target_observation=False,
            totals_uncertainty_modelled=False,
            input_errors_independent_across_contexts=True,
            residual_prior_std=self.config["prior_std"],
            evidence_bindings=self._input_evidence_ids.copy(),
            context_id=self.context.context_id,
            candidates=[list(a) for a in pool],
            channels=channels,
        )
        if "input_feedback_model" in self.config:
            result["joint_input_posterior"] = self.input_posterior().audit()
            result["feedback_used_for_proposals"] = self.config.get(
                "input_feedback_proposals", False
            )
            result["joint_prediction"] = {
                channel: {
                    key: value.tolist()
                    for key, value in self.predict_input_response(
                        pool, ["optimize"] * len(pool), channel
                    ).items()
                }
                for channel in self.config["channels"]
            }
        return result

    def density_response_audit(self, pool):
        """Prediction diagnostics only; physical utility and validity are unchanged."""
        if not self._losses:
            raise ValueError("Density response is disabled")
        slopes = {}
        variance = self.config["density_slope_std"] ** 2
        for channel, rows in self.observations.items():
            active, kinds, _, factor, alpha, _ = self._system(channel)
            z = self._loss_feature(active, [r["context_id"] for r in rows], kinds, channel)
            mean = self.config["density_slope_mean"] + variance * float(z @ alpha)
            posterior_variance = variance
            if len(z):
                solved = np.linalg.solve(factor, variance * z)
                posterior_variance -= float(solved @ solved)
            slopes[channel] = dict(
                mean=mean,
                std=float(np.sqrt(max(posterior_variance, 0.0))),
                observed_quality_count=len(rows),
            )
        loss = self._losses[self.context.context_id]
        return dict(
            model=self.config["density_response_model"],
            normalization="L/active_electrons",
            scale_electrons=loss.scale,
            prior_mean=self.config["density_slope_mean"],
            prior_std=self.config["density_slope_std"],
            slopes=slopes,
            independent_channel_slopes=True,
            prior_is_calibrated=False,
            physical_utility_unchanged=True,
            validity_model_unchanged=True,
            candidates=[dict(active=list(a), **loss.audit(a)) for a in pool],
        )

    def utility_quadratic(self, rng=None):
        weights = self.config["weights"]
        total = sum(weights.values())
        result = Quadratic(
            0.0, np.zeros(self.context.nmo), np.zeros((self.context.nmo, self.context.nmo))
        )
        for channel in self.config["channels"]:
            result = result + self.posterior_quadratic(channel, rng).scaled(
                weights[channel] / total
            )
        return result
