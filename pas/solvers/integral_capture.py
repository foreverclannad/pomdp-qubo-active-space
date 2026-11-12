"""Explicit integral-only adapter at the Molcas -> Block2 execution boundary.

It never impersonates a successful CI solver and never writes energies/RDMs.
Exit 42 intentionally stops the integral-export subprocess after the FCIDUMP
and its binding receipt have been persisted. That exit is not physical evidence.
"""

import os
import shutil
from pathlib import Path

from pas.common import file_hash, write_json


def main():
    destination = Path(os.environ["PAS_INTEGRAL_CAPTURE"]).resolve()
    work = Path.cwd().resolve()
    from pas.backend import checked_directory

    checked_directory(destination, int(os.environ.get("OMP_NUM_THREADS", "1")))
    checked_directory(work, int(os.environ.get("OMP_NUM_THREADS", "1")))
    source = work / "FCIDUMP"
    if not source.is_file() or not (work / "dmrg.conf").is_file():
        raise RuntimeError("Molcas did not supply its Block2 integral/config pair")
    orbitals = work / "INPORB"
    if not orbitals.is_file():
        raise RuntimeError("Molcas native input orbital file is missing at the integral boundary")
    if (destination / "integral_capture.json").exists():
        raise RuntimeError("Integral capture is single use")
    shutil.copy2(source, destination / "FCIDUMP")
    shutil.copy2(work / "dmrg.conf", destination / "export_dmrg.conf")
    shutil.copy2(orbitals, destination / "native_input.InpOrb")
    write_json(
        destination / "integral_capture.json",
        dict(
            schema="pas_molcas_integral_capture_v1",
            source_work_directory=str(work),
            fcidump_sha256=file_hash(source),
            configuration_sha256=file_hash(work / "dmrg.conf"),
            native_input_orbitals_sha256=file_hash(orbitals),
            intended_exit_code=42,
            dmrg_sweeps_executed=0,
            physical_solver_success=False,
        ),
    )
    raise SystemExit(42)


if __name__ == "__main__":
    main()
