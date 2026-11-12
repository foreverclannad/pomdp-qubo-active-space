"""Frozen defaults; changing an effective setting creates a new run."""

from __future__ import annotations

import math
from copy import deepcopy

from .metrics import validate_composition_purity_weight

CHANNELS = ("occupation", "composition", "qit")
DEFAULTS = dict(
    backend="cas",
    channels=["occupation", "composition"],
    weights={"occupation": 1.0, "composition": 1.0},
    diagnostic_budget=6,
    optimization_budget=6,
    seed=20261004,
    pool_size=8,
    annealing_steps=1_000_000,
    annealing_restarts=8,
    exact_limit=100_000,
    fantasies=128,
    prior_mean=0.5,
    prior_std=0.25,
    discrepancy_std=0.25,
    observation_noise=0.0,
    jitter=1e-12,
    max_jitter=1e-8,
    acceptance_tolerance=1e-6,
    voi_tolerance=1e-4,
    threads=32,
    max_macro=200,
    max_super_ci=100,
    thr_energy=1e-6,
    thr_orbital=1e-4,
    thr_ci=1e-4,
    orbital_optimizer="quasi_newton",
    dmrg_probe_m=250,
    dmrg_optimize_m=1000,
    dmrg_max_sweeps=200,
    full_window_survey=True,
    electron_partition_policy="scf_roles_v1",
    composition_model="projection",
)


def configuration(options=None):
    cfg = deepcopy(DEFAULTS)
    supplied = options or {}
    unknown = set(supplied) - (
        set(cfg)
        | {
            "response_model",
            "density_response_model",
            "survey_model",
            "input_response_model",
            "density_slope_mean",
            "density_slope_std",
            "evidence_prior_std",
            "evidence_noise_std",
            "orbital_optimization",
            "proposal_score_model",
            "composition_purity_weight",
            "composition_preferences",
            "input_feedback_model",
            "input_feedback_proposals",
        }
    )
    if unknown:
        raise ValueError(f"Unknown settings: {sorted(unknown)}")
    cfg.update(deepcopy(supplied))
    from .preferences import normalize_preferences

    preferences = normalize_preferences(cfg.pop("composition_preferences", None))
    if preferences is not None:
        if "composition" not in cfg["channels"]:
            raise ValueError("Composition preferences require the composition channel")
        if cfg.get("survey_model") == "correlation_cut_v1":
            raise ValueError("Composition preferences require a composition-aware proposal survey")
        cfg["composition_preferences"] = preferences
    if "orbital_optimization" in cfg:
        if type(cfg["orbital_optimization"]) is not bool:
            raise ValueError("orbital_optimization must be a boolean")
        if cfg["orbital_optimization"]:
            cfg.pop("orbital_optimization")  # Preserve existing optimizing protocol identities.
        elif cfg["backend"] not in ("cas", "dmrg"):
            raise ValueError("Fixed orbital optimization actions require native CAS or DMRG")
    model = cfg.get("response_model", "context_local_v1")
    if model not in ("context_local_v1", "shared_projectors_v1"):
        raise ValueError("Unknown response model")
    if model == "context_local_v1":
        # Preserve the effective identity of pre-sharing frozen configurations.
        cfg.pop("response_model", None)
    if "density_response_model" in cfg:
        if (
            cfg["density_response_model"] != "linear_loss_v1"
            or model != "shared_projectors_v1"
            or cfg["backend"] not in ("cas", "pyscf")
            or "qit" in cfg["channels"]
        ):
            raise ValueError(
                "Density response requires shared CAS/PySCF occupation/composition models"
            )
        for key in ("density_slope_mean", "density_slope_std"):
            if key not in cfg or not math.isfinite(cfg[key]):
                raise ValueError("Density response requires explicit finite slope priors")
        if cfg["density_slope_std"] < 0:
            raise ValueError("Density slope standard deviation cannot be negative")
    elif {"density_slope_mean", "density_slope_std"} & set(cfg):
        raise ValueError("Density slope settings require an enabled density response model")
    channels = cfg["channels"]
    if not channels or len(set(channels)) != len(channels) or not set(channels) <= set(CHANNELS):
        raise ValueError("channels must be a nonempty unique subset of occupation/composition/qit")
    cfg["channels"] = [c for c in CHANNELS if c in channels]
    if cfg["composition_model"] not in (
        "projection",
        "determinant_pair_v1",
        "harmonic_determinant_pair_v1",
        "harmonic_channel_pair_v1",
    ):
        raise ValueError("Unknown composition model")
    if "composition" not in channels:
        # A disabled evidence source cannot alter the effective identity or RNG.
        cfg["composition_model"] = "projection"
        cfg.pop("composition_purity_weight", None)
    if "composition_purity_weight" in cfg:
        alpha = validate_composition_purity_weight(
            cfg["composition_purity_weight"], cfg["composition_model"]
        )
        if alpha == 0.5:
            cfg.pop("composition_purity_weight")  # Preserve legacy identities and random streams.
        else:
            cfg["composition_purity_weight"] = alpha
    if "weights" not in supplied:
        cfg["weights"] = {c: 1.0 for c in channels}
    if set(cfg["weights"]) != set(channels) or any(
        not math.isfinite(v) or v <= 0 for v in cfg["weights"].values()
    ):
        raise ValueError("Provide one positive weight for each enabled channel")
    if cfg["backend"] not in ("cas", "dmrg", "pyscf"):
        raise ValueError("Unknown backend")
    # Opt-in only: do not add a default key to previously frozen identities.
    if "survey_model" in cfg:
        if cfg["survey_model"] not in (
            "correlation_cut_v1",
            "evidence_capture_v1",
            "posterior_evidence_capture_v1",
        ):
            raise ValueError("Unknown survey model")
        if cfg["survey_model"] == "posterior_evidence_capture_v1":
            expected = CHANNELS if cfg["backend"] == "dmrg" else ("occupation", "composition")
            if cfg["backend"] not in ("cas", "dmrg") or set(channels) != set(expected):
                raise ValueError(
                    "Posterior capture requires CAS occupation/composition or all three DMRG channels"
                )
        else:
            if cfg["backend"] != "dmrg" or "qit" not in channels:
                raise ValueError("Explicit survey models require DMRG and QIT evidence")
            if cfg["survey_model"] == "evidence_capture_v1" and set(channels) != set(CHANNELS):
                raise ValueError(
                    "Evidence capture requires occupation, composition and QIT evidence"
                )
        if (
            cfg["survey_model"] == "evidence_capture_v1"
            and cfg["composition_model"] != "projection"
        ):
            raise ValueError("Raw evidence capture supports projection composition only")
    if cfg.get("survey_model") == "posterior_evidence_capture_v1":
        if not cfg["full_window_survey"]:
            raise ValueError("Posterior capture requires per-basis surveys")
        for key in ("evidence_prior_std", "evidence_noise_std"):
            value = cfg.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(
                    "Posterior capture needs explicit positive evidence prior/noise scales"
                )
            if not math.isfinite(value * value) or value * value == 0:
                raise ValueError("Evidence variances must be representable positive finite numbers")
    elif {"evidence_prior_std", "evidence_noise_std"} & set(cfg):
        raise ValueError("Evidence error settings require posterior capture")
    if "input_response_model" in cfg:
        if (
            cfg["input_response_model"] != "posterior_descriptor_residual_v1"
            or cfg.get("survey_model") != "posterior_evidence_capture_v1"
            or "density_response_model" in cfg
        ):
            raise ValueError(
                "Input response requires posterior capture and its descriptor residual model"
            )
    if "input_feedback_model" in cfg:
        if (
            cfg["input_feedback_model"] != "joint_gaussian_v1"
            or cfg.get("input_response_model") != "posterior_descriptor_residual_v1"
        ):
            raise ValueError("Joint input feedback requires the input descriptor residual model")
    if "input_feedback_proposals" in cfg:
        if type(cfg["input_feedback_proposals"]) is not bool:
            raise ValueError("input_feedback_proposals must be a boolean")
        if cfg["input_feedback_proposals"]:
            if "input_feedback_model" not in cfg:
                raise ValueError("Feedback proposals require joint input feedback")
        else:
            cfg.pop("input_feedback_proposals")
    if "proposal_score_model" in cfg:
        if (
            cfg["proposal_score_model"] != "terminal_aligned_v1"
            or cfg.get("survey_model") != "posterior_evidence_capture_v1"
            or cfg.get("input_response_model") != "posterior_descriptor_residual_v1"
        ):
            raise ValueError(
                "Terminal-aligned proposals require posterior capture and exact input descriptors"
            )
    if cfg["electron_partition_policy"] != "scf_roles_v1":
        raise ValueError("SCF electron roles are required for candidate partitions")
    if cfg["backend"] == "cas" and "qit" in channels:
        raise ValueError(
            "Native OpenMolcas CAS QIT extraction is not supported; no implicit solver fallback"
        )
    for key in (
        "diagnostic_budget",
        "optimization_budget",
        "annealing_steps",
        "annealing_restarts",
        "exact_limit",
        "fantasies",
        "pool_size",
        "threads",
        "max_macro",
        "max_super_ci",
        "dmrg_probe_m",
        "dmrg_optimize_m",
        "dmrg_max_sweeps",
    ):
        if type(cfg[key]) is not int or cfg[key] < (0 if "budget" in key else 1):
            raise ValueError(f"Invalid {key}")
    if cfg["threads"] > 32 or cfg["orbital_optimizer"] != "quasi_newton":
        raise ValueError("At most 32 threads and QN orbital optimization are required")
    if cfg["annealing_steps"] % cfg["annealing_restarts"] or cfg["fantasies"] % 2:
        raise ValueError("Annealing steps must divide evenly over restarts; fantasies must be even")
    if cfg["observation_noise"] != 0:
        raise ValueError("Frozen protocol observations have zero observation noise")
    for key in (
        "prior_std",
        "discrepancy_std",
        "jitter",
        "max_jitter",
        "thr_energy",
        "thr_orbital",
        "thr_ci",
    ):
        if not math.isfinite(cfg[key]) or cfg[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    for key in ("prior_mean", "acceptance_tolerance", "voi_tolerance"):
        if not math.isfinite(cfg[key]) or not 0 <= cfg[key] <= 1:
            raise ValueError(f"{key} must be finite and in [0,1]")
    if not 1e-12 <= cfg["jitter"] <= cfg["max_jitter"] <= 1e-8:
        raise ValueError("Numerical jitter must remain in the fixed [1e-12, 1e-8] interval")
    if type(cfg["full_window_survey"]) is not bool:
        raise ValueError("full_window_survey must be a boolean")
    if cfg.get("survey_model") != "posterior_evidence_capture_v1" and (
        cfg["backend"] != "dmrg" or not ({"occupation", "qit"} & set(channels))
    ):
        cfg["full_window_survey"] = False
    return cfg


def planning_configuration(config, response_policy=None):
    """Derive only the response-model opt-in; retain physical configuration IDs.

    The policy belongs to the run manifest, not to immutable physical requests.
    Without it, legacy configurations and configurations already carrying the
    explicit model key retain their exact effective values.
    """
    cfg = configuration(config)
    if response_policy is None:
        return cfg
    expected = dict(schema="pas_input_response_policy_v1", model="posterior_descriptor_residual_v1")
    if (
        not isinstance(response_policy, dict)
        or {k: v for k, v in response_policy.items() if k != "input_feedback"} != expected
    ):
        raise ValueError("Unsupported response policy; provide its exact schema and model")
    if cfg.get("input_response_model", response_policy["model"]) != response_policy["model"]:
        raise ValueError("Response policy differs from the existing input response model")
    cfg["input_response_model"] = response_policy["model"]
    if "input_feedback" in response_policy:
        feedback = response_policy["input_feedback"]
        if (
            not isinstance(feedback, dict)
            or set(feedback) != {"model", "proposals"}
            or feedback["model"] != "joint_gaussian_v1"
            or type(feedback["proposals"]) is not bool
        ):
            raise ValueError("Unsupported joint input feedback policy")
        if cfg.get("input_feedback_model", feedback["model"]) != feedback["model"] or (
            "input_feedback_proposals" in cfg
            and cfg["input_feedback_proposals"] != feedback["proposals"]
        ):
            raise ValueError("Input feedback policy differs from the existing configuration")
        cfg.update(
            input_feedback_model=feedback["model"], input_feedback_proposals=feedback["proposals"]
        )
    return configuration(cfg)


def validate_production_plan(options=None):
    """Guard the agreed four-point protocol; smoke tests may use smaller settings.

    Channel masks and positive fixed weights remain configurable. A campaign
    cannot silently substitute smaller QUBO budgets or different priors/protocols.
    """
    cfg = configuration(options)
    if cfg["backend"] not in ("cas", "dmrg"):
        raise ValueError("Production requires native CAS or the DMRG backend")
    fixed = (
        "seed",
        "pool_size",
        "annealing_steps",
        "annealing_restarts",
        "exact_limit",
        "fantasies",
        "prior_mean",
        "prior_std",
        "discrepancy_std",
        "observation_noise",
        "jitter",
        "max_jitter",
        "acceptance_tolerance",
        "voi_tolerance",
        "max_macro",
        "max_super_ci",
        "thr_energy",
        "thr_orbital",
        "thr_ci",
        "orbital_optimizer",
        "dmrg_probe_m",
        "dmrg_optimize_m",
        "dmrg_max_sweeps",
        "electron_partition_policy",
    )
    mismatches = {
        key: dict(required=DEFAULTS[key], actual=cfg[key])
        for key in fixed
        if cfg[key] != DEFAULTS[key]
    }
    if mismatches:
        raise ValueError(f"Configuration differs from the frozen production plan: {mismatches}")
    if cfg["diagnostic_budget"] > 6 or cfg["optimization_budget"] > 6:
        raise ValueError(
            "Production permits at most six diagnostics and six optimizations per point"
        )
    expected_survey = (
        cfg.get("survey_model") == "posterior_evidence_capture_v1"
        or cfg["backend"] == "dmrg"
        and bool(set(cfg["channels"]) & {"occupation", "qit"})
    )
    if cfg["full_window_survey"] != expected_survey:
        raise ValueError("Production DMRG initializes with one whole-window evidence survey")
    return cfg
