"""Continue native Block2 sweeps until the diagnostic's state checks agree.

Uses DMRG.solve(..., sweep_start=...) on the same MPS and native sweep history.
The stage schedule and total sweep budget are unchanged. The installed driver
is compiled with a narrow AST call substitution; no shared file is edited.
"""

from __future__ import annotations

import ast
import importlib.metadata
import sys
from functools import partial
from pathlib import Path

import numpy as np

from pas.common import file_hash, read_json, write_json


def assessment(energies, nroots, threshold, native_tol, *, same_direction=True):
    a = np.asarray(energies, dtype=float)
    result = dict(
        passed=False, sweeps=len(a), all_state_threshold=threshold, native_tolerance=native_tol
    )
    minimum = 4 if same_direction else 2
    if a.ndim != 2 or a.shape[1] != nroots or len(a) < minimum or not np.all(np.isfinite(a)):
        return {**result, "reason": "insufficient_complete_state_sweeps"}
    adjacent = float(np.max(np.abs(a[-1] - a[-2])))
    same = float(np.max(np.abs(a[-2:] - a[-4:-2]))) if len(a) >= 4 else None
    last_root = float(abs(a[-1, -1] - a[-2, -1]))
    return dict(
        result,
        passed=adjacent <= threshold
        and (not same_direction or same <= threshold)
        and last_root < native_tol,
        adjacent_max_state_delta=adjacent,
        same_direction_max_state_delta=same,
        native_last_root_delta=last_root,
        reason="state_sweep_check",
    )


def checked_solve(dmrg, n_sweeps, forward, tol, *, nroots, threshold, audit, same_direction=True):
    if tol <= 0:
        raise ValueError("Fixed warmup stages must use the original native solve")
    energy = dmrg.solve(n_sweeps, forward, tol)
    calls = 1
    while True:
        before = np.asarray(dmrg.energies, dtype=float).copy()
        check = assessment(before, nroots, threshold, tol, same_direction=same_direction)
        count = len(before)
        if check["passed"] or count >= n_sweeps:
            break
        # Do not turn an abort/non-convergence return into a continuation.
        if count < 2 or before.ndim != 2 or abs(before[-1, -1] - before[-2, -1]) >= tol:
            check["reason"] = "native_did_not_stop_on_energy_tolerance"
            break
        energy = dmrg.solve(n_sweeps, dmrg.forward, tol, sweep_start=count)
        calls += 1
        after = np.asarray(dmrg.energies, dtype=float)
        if (
            len(after) <= count
            or len(after) > n_sweeps
            or not np.array_equal(after[:count], before)
        ):
            raise RuntimeError("Native continuation changed sweep history or exceeded its budget")
    audit.append(dict(check, native_solve_calls=calls, stage_sweep_limit=n_sweeps))
    print("[PAS_QIT_SWEEP_GATE]", audit[-1], flush=True)
    return energy


def instrument_main(source, filename):
    class Calls(ast.NodeTransformer):
        count = 0

        def visit_Call(self, node):
            self.generic_visit(node)
            if (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "dmrg"
                and node.func.attr == "solve"
                and len(node.args) == 3
                and isinstance(node.args[2], ast.Name)
                and node.args[2].id == "sweep_tol"
            ):
                self.count += 1
                node.func = ast.Name(id="_pas_solve", ctx=ast.Load())
                node.args.insert(0, ast.Name(id="dmrg", ctx=ast.Load()))
                node.keywords.append(
                    ast.keyword(arg="nroots", value=ast.Name(id="nroots", ctx=ast.Load()))
                )
            return node

    tree = ast.parse(source, filename)
    transform = Calls()
    tree = transform.visit(tree)
    return compile(ast.fix_missing_locations(tree), filename, "exec"), transform.count


def main():
    from pas.backend import BLOCK2MAIN

    if importlib.metadata.version("block2") != "0.5.3":
        raise RuntimeError("Unverified Block2 version")
    path = Path(BLOCK2MAIN)
    code, count = instrument_main(path.read_text(), str(path))
    if count != 5:
        raise RuntimeError("Pinned Block2 solve call sites changed")
    audit = []
    sys.argv = [str(path), *sys.argv[1:]]
    binding = Path("state_sweep_gate_request.json")
    request = read_json(binding) if binding.exists() else {}
    from .block2_convergence import settings, validate_fixed_policy

    policy = validate_fixed_policy(request.get("fixed_dmrg_policy"))
    extra = {}
    numerical_failure = None
    if policy is None:
        threshold = settings(request.get("profile", "standard"))["all_state_delta_tolerance"]
        solve = partial(checked_solve, threshold=threshold, audit=audit, same_direction=False)
    else:
        # Import inside this branch: molcas_sweep_endpoint itself uses instrument_main.
        from pyblock2.driver.parser import parse

        from .molcas_sweep_endpoint import NumericalNonconvergence, endpoint_solve

        native = request.get("native_config", {})
        if (
            len(sys.argv) != 2
            or file_hash(sys.argv[1]) != native.get("effective_input_sha256")
            or not native.get("parser_roundtrip_verified")
        ):
            raise ValueError("Native diagnostic gate configuration differs from its bound input")
        actual = parse(sys.argv[1])
        if (
            float(actual["sweep_tol"]) != native["sweep_tol"]
            or int(actual["maxiter"]) != native["maxiter"]
            or actual["schedule"] != native["schedule"]
            or int(actual["twodot_to_onedot"]) != native["twodot_to_onedot"]
        ):
            raise ValueError("Native diagnostic gate settings differ from the resolved schedule")
        solve = partial(endpoint_solve, audit=audit)
        numerical_failure = NumericalNonconvergence
        extra = dict(
            fixed_dmrg_policy=policy,
            native_config=native,
            convergence_rule="all_roots_strict_native_tolerance_at_center_zero",
            thresholds_changed=False,
        )
    namespace = dict(__name__="__main__", __file__=str(path), __package__=None, _pas_solve=solve)
    try:
        exec(code, namespace)
    except Exception as exc:
        if numerical_failure is not None and isinstance(exc, numerical_failure):
            raise SystemExit(96) from exc
        raise
    finally:
        write_json(
            Path("state_sweep_gate.json"),
            dict(
                schema="pas_state_sweep_gate_v1",
                block2_version="0.5.3",
                block2main_sha256=file_hash(path),
                instrumented_calls=count,
                stages=audit,
                passed=bool(audit) and all(x["passed"] for x in audit),
                installed_driver_modified=False,
                **extra,
            ),
        )


if __name__ == "__main__":
    main()
