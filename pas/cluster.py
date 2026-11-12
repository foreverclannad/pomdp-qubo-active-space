"""Bind the executable files and libraries used by a native physical protocol."""

from __future__ import annotations

import os
import shlex
import shutil
from pathlib import Path

from .common import file_hash


def runtime_bindings(config):
    """Freeze the actual executable files used by this solver protocol."""
    import sys
    from importlib.metadata import version
    from importlib.util import find_spec

    from . import backend

    paths = [Path(sys.executable).resolve()]
    modules = []
    module_prefix = None
    if config["backend"] == "pyscf":
        modules += [
            "pyscf",
            "pyscf.scf.hf",
            "pyscf.mcscf.mc1step",
            "pyscf.mcscf.casci",
            "pyscf.mcscf.addons",
            "pyscf.fci.direct_spin0",
            "pyscf.fci.direct_spin1",
        ]
    else:
        base = Path(backend.MOLCAS if config["backend"] == "dmrg" else backend.CAS_MOLCAS)
        pymolcas = base / "pymolcas"
        # The pinned pymolcas embeds its Python modules in this script. Its
        # env-python shebang can resolve to a different binary than this process.
        with pymolcas.open() as source:
            shebang = source.readline().strip().removeprefix("#!").split()
        if len(shebang) == 2 and Path(shebang[0]).name == "env":
            interpreter = shutil.which(shebang[1])
        elif len(shebang) == 1 and Path(shebang[0]).is_absolute():
            interpreter = shebang[0]
        else:
            raise ValueError("Unrecognized pymolcas interpreter binding")
        if interpreter is None:
            raise ValueError("pymolcas interpreter is unavailable")
        paths += [pymolcas, Path(interpreter)]
        paths += [
            base / "bin" / name for name in ("gateway.exe", "seward.exe", "scf.exe", "rasscf.exe")
        ]
        if os.environ.get("MOLCAS_DEBUGGER"):
            argv = shlex.split(os.environ["MOLCAS_DEBUGGER"])
            if not argv or not Path(argv[0]).is_absolute():
                raise ValueError("Molcas module prefix requires an absolute executable")
            prefix_files = [
                Path(arg) for i, arg in enumerate(argv) if i == 0 or not arg.startswith("-")
            ]
            if any(not p.is_absolute() or not p.is_file() for p in prefix_files):
                raise ValueError("Molcas module prefix arguments must name absolute existing files")
            paths += prefix_files
            module_prefix = dict(
                argv=argv,
                environment={"OMP_MAX_ACTIVE_LEVELS": os.environ.get("OMP_MAX_ACTIVE_LEVELS")},
            )
            if "PAS_RELEASE_AO_INTEGRALS" in os.environ:
                module_prefix["environment"]["PAS_RELEASE_AO_INTEGRALS"] = os.environ[
                    "PAS_RELEASE_AO_INTEGRALS"
                ]
    packages = {name: version(name) for name in ("numpy", "scipy", "h5py", "pyscf")}
    if config["backend"] == "dmrg":
        import block2

        if version("block2") != "0.5.3":
            raise ValueError("Production requires Block2 0.5.3")
        paths += [Path(backend.BLOCK2MAIN), Path(block2.__file__)]
        modules += ["pyblock2.driver.core", "pyblock2.driver.parser"]
        packages["block2"] = "0.5.3"
    for name in modules:
        spec = find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError("Physical driver source is unavailable: " + name)
        paths.append(Path(spec.origin))
    result = dict(
        backend=config["backend"],
        packages=packages,
        files=[
            dict(path=str(p), sha256=file_hash(p)) for p in sorted({p.resolve() for p in paths})
        ],
    )
    if module_prefix is not None:
        result["module_prefix"] = module_prefix
    return result
