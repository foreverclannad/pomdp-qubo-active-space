"""Read-only checks of the unchanged native cu07 execution restrictions."""

import os
import socket
from pathlib import Path

from pas import backend


class EnvironmentError(RuntimeError):
    """The requested native execution cannot run in this environment."""


def preflight(directory=None, threads=32, backend_name="dmrg"):
    configured = Path(os.environ.get("PAS_WORK_ROOT", "/tmp"))
    root = configured.resolve()
    paths = {"molcas": backend.CAS_MOLCAS if backend_name == "cas" else backend.MOLCAS}
    if backend_name == "dmrg":
        paths["block2main"] = backend.BLOCK2MAIN
    checks = {
        "host_is_cu07": socket.gethostname().split(".")[0] == "cu07",
        "threads_in_1_to_32": type(threads) is int and 1 <= threads <= 32,
        "work_root_absolute": configured.is_absolute(),
        "work_root_exists": root.is_dir(),
        "work_root_not_filesystem_root": root != Path(root.anchor),
        "work_root_writable": os.access(root, os.W_OK),
        "work_root_owned": root == Path("/tmp")
        or (root.exists() and hasattr(os, "getuid") and root.stat().st_uid == os.getuid()),
    }
    if directory is not None:
        target = Path(directory).resolve()
        checks["output_under_work_root"] = target.is_relative_to(root) and target != root
    checks.update({name + "_exists": Path(path).exists() for name, path in paths.items()})
    checks["pymolcas_exists"] = (Path(paths["molcas"]) / "pymolcas").is_file()
    return dict(
        passed=all(checks.values()),
        host=socket.gethostname(),
        threads=threads,
        work_root=str(root),
        required_paths=paths,
        checks=checks,
        scope="Static environment check; no solver execution",
    )


def require_cluster(directory, config):
    report = preflight(directory, config["threads"], config["backend"])
    if not report["passed"]:
        missing = ", ".join(name for name, ok in report["checks"].items() if not ok)
        raise EnvironmentError("Native execution requirements not met: " + missing)
    return report
