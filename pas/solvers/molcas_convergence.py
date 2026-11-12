"""Task-local Block2 input overrides, preserving the pinned physical runtime."""

from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from pas.common import file_hash, write_json


def standard_input(parsed, maxiter=150, *, profile="standard"):
    """Transform the *resolved* 0.5.3 schedule, including schedule default.

    Preserve the pinned native standard CI tolerances and noise schedule.
    Only the explicit total sweep allowance is set; no alternate profile exists.
    """
    if int(maxiter) != maxiter or not 14 <= maxiter <= 200:
        raise ValueError("Block2 sweep limit must be in [14, 200]")
    if profile != "standard":
        raise ValueError("Only the standard frozen protocol is supported")
    data = copy.deepcopy(parsed)
    schedule = data.pop("schedule")
    if len(schedule) != 4 or any(schedule[3]):
        raise ValueError("Expected the pinned Molcas scalar bond-dimension schedule")
    n = int(maxiter)
    if len(schedule[0]) > n:
        raise ValueError("Native schedule exceeds frozen sweep allowance")
    for values in schedule:
        if not values:
            raise ValueError("Empty resolved schedule")
        values.extend([copy.deepcopy(values[-1]) for _ in range(n - len(values))])
    data["maxiter"] = str(n)
    data["sweep_tol"] = format(float(data.get("sweep_tol", 1e-6)), ".12g")
    lines = [f"{key} {value}".rstrip() for key, value in data.items()]
    lines.append("schedule")
    previous = None
    for i, (bond, tol, noise) in enumerate(zip(*schedule[:3])):
        row = (bond, tol, noise)
        if row != previous:
            lines.append(f"{i} {bond} {tol:.12g} {noise:.12g}")
        previous = row
    lines.append("end")
    return "\n".join(lines) + "\n", dict(
        sweep_tol=float(data["sweep_tol"]),
        maxiter=n,
        schedule=schedule,
        thresholds_changed=False,
        noise_and_bond_dimensions_preserved=True,
    )


def install(directory, environment, maxiter):
    directory = Path(directory).resolve()
    bindir = directory / "convergence_bin"
    bindir.mkdir(exist_ok=True)
    wrapper = bindir / "block2main"
    wrapper.write_text(
        "#!" + sys.executable + "\nfrom pas.solvers.molcas_convergence import main\nmain()\n"
    )
    wrapper.chmod(0o700)
    return {
        **environment,
        "PATH": str(bindir) + os.pathsep + environment["PATH"],
        "PYTHONPATH": str(Path(__file__).resolve().parents[2])
        + os.pathsep
        + os.environ.get("PYTHONPATH", ""),
        "PAS_BLOCK2_CONVERGENCE_ROOT": str(directory),
        "PAS_BLOCK2_MAXITER": str(int(maxiter)),
    }


def run_with_density_energy(path, call, nroots):
    # The wrapper waits for its child, so Molcas's process CPU accounting still
    # includes the DMRG execution. No extra optimization/validation is launched.
    from .molcas_density_energy import reconcile

    result = subprocess.run(
        [sys.executable, "-m", "pas.solvers.molcas_sweep_endpoint", str(path)],
        env=dict(os.environ, PAS_BLOCK2_CALL_AUDIT=str(Path(call).resolve())),
    )
    write_json(
        Path(call) / "process.json",
        dict(
            return_code=result.returncode,
            numerical_endpoint_exit_code=96,
            external_wrapper="pas.solvers.molcas_sweep_endpoint",
        ),
    )
    if result.returncode:
        raise SystemExit(result.returncode)
    return reconcile(Path(path).parent, call, nroots)


def main():
    from pyblock2.driver.parser import parse

    from pas.backend import runtime

    runtime()
    root = Path(os.environ["PAS_BLOCK2_CONVERGENCE_ROOT"]).resolve()
    if len(sys.argv) != 2:
        raise ValueError("Expected one Molcas-generated input")
    path = Path(sys.argv[1]).resolve()
    if not path.is_relative_to(root / "work"):
        raise ValueError("Override is restricted to this task working directory")
    audit = root / "block2_calls"
    audit.mkdir(exist_ok=True)
    index = len(list(audit.glob("call_*"))) + 1
    call = audit / f"call_{index:04d}"
    call.mkdir(exist_ok=False)
    original = path.read_text()
    (call / "original.conf").write_text(original)
    before = parse(str(path))
    text, expected = standard_input(before, int(os.environ["PAS_BLOCK2_MAXITER"]))
    effective = call / "effective.conf"
    effective.write_text(text)
    # Full parser round trip before executing: assert the generated tolerances,
    # electron/state/reorder settings, noise and bond dimensions are what we intend.
    after = parse(str(effective))
    for key, value in before.items():
        if key not in ("schedule", "sweep_tol", "maxiter") and after[key] != value:
            raise ValueError("Physical setting changed: " + key)
    for actual, wanted in zip(after["schedule"], expected["schedule"]):
        np.testing.assert_allclose(actual, wanted, atol=0, rtol=1e-12)
    if float(after["sweep_tol"]) != expected["sweep_tol"]:
        raise ValueError("Effective sweep tolerance changed")
    write_json(
        call / "audit.json",
        dict(
            schema="pas_molcas_block2_convergence_v1",
            created_unix=time.time(),
            profile="standard",
            original_sha256=file_hash(call / "original.conf"),
            effective_sha256=file_hash(effective),
            runtime=runtime(),
            settings=expected,
            original_sweep_tol=float(before.get("sweep_tol", 1e-6)),
            parser_roundtrip_verified=True,
        ),
    )
    path.write_text(text)
    run_with_density_energy(path, call, int(after["nroots"]))
