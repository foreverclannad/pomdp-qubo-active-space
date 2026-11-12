"""Finish each Molcas DMRG call at the same one-site sweep endpoint.

The native tolerance, schedules and total sweep budget are preserved. A final
half sweep can remove the alternating left/right endpoint of warm restarts.
Every state must satisfy the native tolerance at that exact final endpoint.
"""

import os
import sys
from functools import partial
from pathlib import Path

import numpy as np

from pas.common import file_hash, write_json

from .diagnostic_sweep_gate import instrument_main


class NumericalNonconvergence(RuntimeError):
    pass


def endpoint_solve(dmrg, n_sweeps, forward, tol, *, nroots, audit):
    if tol <= 0:
        raise ValueError("Fixed warmup stages must use native solve")
    if dmrg.me.dot != 1 or dmrg.sweep_start_site != 0:
        raise ValueError("Only the full-window one-site final stage is verified")
    record = dict(
        passed=False,
        stage_sweep_limit=n_sweeps,
        native_tolerance=tol,
        target_center=0,
        extra_half_sweeps=0,
    )
    try:
        energy = dmrg.solve(n_sweeps, forward, tol)
        initial = len(dmrg.energies)
        record["initial_native_sweeps"] = initial
        while True:
            before = np.asarray(dmrg.energies, dtype=float).copy()
            count = len(before)
            if (
                before.ndim != 2
                or before.shape[1] != nroots
                or count < 2
                or count > n_sweeps
                or not np.all(np.isfinite(before))
            ):
                raise RuntimeError("Incomplete or invalid native sweep history")
            deltas = np.abs(before[-1] - before[-2])
            delta = float(np.max(deltas))
            record.update(
                sweeps=count,
                final_center=int(dmrg.me.ket.center),
                all_root_deltas=deltas.tolist(),
                all_root_max_delta=delta,
                extra_half_sweeps=count - initial,
            )
            if dmrg.me.ket.center == 0 and delta < tol:
                record["passed"] = True
                return energy
            if count >= n_sweeps:
                raise NumericalNonconvergence(
                    "Sweep budget exhausted before all-root consistent endpoint"
                )
            # Continue just one half sweep so the native minimum sweep count
            # cannot repeatedly skip the required endpoint. The saved history,
            # schedule index and remaining global allowance are unchanged.
            energy = dmrg.solve(count + 1, dmrg.forward, tol, sweep_start=count)
            after = np.asarray(dmrg.energies, dtype=float)
            if len(after) != count + 1 or not np.array_equal(after[:count], before):
                raise RuntimeError("Endpoint continuation changed native sweep history")
    except BaseException as exc:
        record.update(passed=False, failure=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        audit.append(record)
        print("[PAS_MOLCAS_SWEEP_ENDPOINT]", record, flush=True)


def main():
    from pas.backend import BLOCK2MAIN, runtime

    runtime()
    path = Path(BLOCK2MAIN)
    call = Path(os.environ["PAS_BLOCK2_CALL_AUDIT"]).resolve()
    root = Path(os.environ["PAS_BLOCK2_CONVERGENCE_ROOT"]).resolve()
    if not call.is_relative_to(root / "block2_calls") or not call.is_dir():
        raise ValueError("Endpoint audit must belong to this physical task")
    code, count = instrument_main(path.read_text(), str(path))
    if count != 5:
        raise RuntimeError("Pinned Block2 solve call sites changed")
    audit = []
    sys.argv = [str(path), *sys.argv[1:]]
    namespace = dict(
        __name__="__main__",
        __file__=str(path),
        __package__=None,
        _pas_solve=partial(endpoint_solve, audit=audit),
    )
    try:
        exec(code, namespace)
    except NumericalNonconvergence:
        raise SystemExit(96)
    finally:
        write_json(
            call / "sweep_endpoint.json",
            dict(
                schema="pas_molcas_sweep_endpoint_v1",
                block2main_sha256=file_hash(path),
                instrumented_calls=count,
                stages=audit,
                passed=bool(audit) and all(x["passed"] for x in audit),
                installed_driver_modified=False,
                thresholds_changed=False,
            ),
        )


if __name__ == "__main__":
    main()
