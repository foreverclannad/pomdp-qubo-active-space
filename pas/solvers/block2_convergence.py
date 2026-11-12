"""Legacy fixed-MO protocol and an explicit native-standard opt-in."""

NATIVE_STANDARD_POLICY = dict(schema="pas_fixed_dmrg_policy_v1", model="molcas_native_standard_v1")


def validate_fixed_policy(policy):
    if policy is None:
        return None
    if not isinstance(policy, dict) or policy != NATIVE_STANDARD_POLICY:
        raise ValueError("Unsupported fixed DMRG policy")
    return dict(policy)


def native_diagnostic_config(native_path, ctx, fidelity, maxiter, threads):
    """Resolve the captured Molcas cold input with the same parser as optimization.

    In block2 0.5.3, default startM=maxM=250 resolves to 58 schedule entries
    although native maxiter is 50. Parse that original input first, then use the
    shared standard_input override to set the total allowance. Do not generate
    default+maxiter=200 first (which resolves to 208 entries for M250).
    """
    import hashlib
    import re
    from importlib.metadata import version
    from pathlib import Path
    from tempfile import TemporaryDirectory

    import numpy as np
    from pyblock2.driver import parser

    from pas.common import file_hash

    from .molcas_convergence import standard_input

    if version("block2") != "0.5.3":
        raise ValueError("Native diagnostic configuration requires block2 0.5.3")
    if type(fidelity) is not int or fidelity not in (250, 1000):
        raise ValueError("Unsupported native diagnostic bond dimension")
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("Threads must be an integer in [1, 32]")
    path = Path(native_path)
    original_hash = file_hash(path)
    original = path.read_text()
    if not re.search(r"^\s*schedule\s+default\s*$", original, re.M):
        raise ValueError("Native diagnostic requires an unmodified cold default schedule")
    before = parser.parse(str(path))
    if (
        any("restart" in key for key in before)
        or "read_mps_tags" in before
        or before.get("warmup") != "local_4site"
        or "openmolcas" not in before
        or before.get("gaopt") != "default"
        or before.get("hf_occ") != "integral"
    ):
        raise ValueError("Native diagnostic requires the Molcas cold initialization")
    if int(before.get("maxiter", -1)) != 50 or int(before.get("maxM", -1)) != fidelity:
        raise ValueError("Native diagnostic maxiter or maxM differs from its captured request")
    states = ctx["states"]
    nroots = len(states["ids"])
    if nroots < 1 or states.get("roots") != list(range(1, nroots + 1)):
        raise ValueError("Native diagnostic requires the declared consecutive lowest roots")
    weights = np.asarray(states["weights"], float)
    actual_weights = np.array([float(x) for x in before.get("weights", "").split()])
    if not actual_weights.size:
        actual_weights = np.full(nroots, 1.0 / nroots)
    if (
        int(before.get("nelec", -1)) != ctx["electrons"]
        or int(before.get("spin", -1)) != ctx["spin"]
        or int(before.get("nroots", -1)) != nroots
        or weights.shape != (nroots,)
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0)
        or not np.isclose(weights.sum(), 1.0, atol=1e-12, rtol=0)
        or actual_weights.shape != weights.shape
        or not np.allclose(actual_weights, weights, atol=1e-12, rtol=0)
    ):
        raise ValueError("Native diagnostic electron, spin, roots or state weights differ")
    if (
        max(before["schedule"][0]) != fidelity
        or before.get("sym", "").lower() != "c1"
        or int(before.get("irrep", -1)) != 1
    ):
        raise ValueError("Native diagnostic resolved dimension or symmetry differs")
    # These are resource controls only. Every scientific key comes from Molcas.
    controlled = dict(before, num_thrds=str(threads), mkl_thrds="1", mem="10g")
    text, audit = standard_input(controlled, maxiter)
    with TemporaryDirectory(prefix="pas-native-config-") as directory:
        effective = Path(directory) / "effective.conf"
        effective.write_text(text)
        after = parser.parse(str(effective))
    for key, value in controlled.items():
        if key not in ("schedule", "maxiter", "sweep_tol") and after[key] != value:
            raise ValueError("Native diagnostic parser changed a physical setting: " + key)
    for actual, expected in zip(after["schedule"], audit["schedule"], strict=True):
        np.testing.assert_allclose(actual, expected, atol=0, rtol=1e-12)
    if float(after["sweep_tol"]) != audit["sweep_tol"] or int(after["maxiter"]) != maxiter:
        raise ValueError("Native diagnostic parser changed the convergence settings")
    if file_hash(path) != original_hash:
        raise ValueError("Captured native diagnostic input changed during parsing")
    audit.update(
        native_input=dict(path=str(path.resolve()), sha256=original_hash),
        effective_input_sha256=hashlib.sha256(text.encode()).hexdigest(),
        parser_sha256=file_hash(parser.__file__),
        block2_version="0.5.3",
        native_maxiter=int(before["maxiter"]),
        native_resolved_schedule_length=len(before["schedule"][0]),
        default_startM_equals_maxM=int(before.get("startM", 250)) == fidelity,
        twodot_to_onedot=int(after["twodot_to_onedot"]),
        initialization="native_cold_local_4site",
        resource_overrides=dict(num_thrds=threads, mkl_thrds=1, memory_gb=10),
        state_weights=actual_weights.tolist(),
        parser_roundtrip_verified=True,
        roundtrip_relative_tolerance=1e-12,
    )
    # Persist the re-parsed effective schedule, including harmless decimal serialization rounding.
    audit["schedule"] = after["schedule"]
    return text, audit


def settings(profile="standard", maxiter=200):
    if profile != "standard":
        raise ValueError("Only the fixed standard protocol is supported")
    if type(maxiter) is not int or not 14 <= maxiter <= 200:
        raise ValueError("Sweep budget must be in [14, 200]")
    return dict(profile="standard", sweep_tol=1e-8, all_state_delta_tolerance=1e-8, maxiter=maxiter)


def diagnostic_config(ctx, fidelity, profile="standard", maxiter=200, *, threads):
    p = settings(profile, maxiter)
    if type(threads) is not int or not 1 <= threads <= 32:
        raise ValueError("Threads must be an integer in [1, 32]")
    return (
        f"sym c1\norbitals FCIDUMP\nnelec {ctx['electrons']}\nnroots {len(ctx['states']['ids'])}\n"
        f"weights {' '.join(map(str, ctx['states']['weights']))}\nspin {ctx['spin']}\nirrep 1\n"
        f"num_thrds {threads}\nmkl_thrds 1\n"
        f"gaopt default\nwarmup local_4site\nmaxiter {p['maxiter']}\nsweep_tol 1e-8\n"
        f"memory 10g\nschedule\n0 {fidelity} 1e-6 1e-5\n"
        f"4 {fidelity} 1e-7 1e-6\n8 {fidelity} 1e-8 0\nend\n"
        "twodot_to_onedot 12\nhf_occ integral\n"
    )
