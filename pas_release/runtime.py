"""Explicit new-run and resume lifecycle under the paper's physical protocol."""

import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path

from pas import backend, controller
from pas.common import identity

from . import engine
from .cluster import require_cluster
from .common import interface_identity, read_json, require, sha, verify_source, write_new
from .inputs import load_input, verify_resolved
from .outputs import state_result, write_results


@contextmanager
def run_lock(directory):
    import fcntl

    with (Path(directory) / ".interface.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def _export(directory, spec, interface_id):
    _, state = engine.verify_run(directory / "engine")
    result, history = state_result(state, spec, interface_id, directory / "engine")
    write_results(directory, result, history)
    return result


def _failure(directory, spec, error):
    write_results(
        directory,
        dict(
            schema="pas.result.v1",
            mode="failure",
            name=spec["name"],
            status="failed",
            error=str(error),
            engine_records="engine/",
        ),
    )


def start(input_path, directory, *, prepare_only=False):
    verify_source()
    spec = load_input(input_path)
    directory = Path(directory).resolve()
    require(
        not directory.exists(), "Output already exists; use pas resume or choose a new directory"
    )
    require(
        not prepare_only or "seed_file" in spec["files"],
        "--prepare-only requires a complete seed_file",
    )
    require_cluster(directory, spec["config"])
    # Save the same input snapshot that produced the resolved configuration.
    original = spec["input_file"]
    raw = read_json(original["path"], expected_sha256=original["sha256"])
    verify_resolved(spec)
    directory.mkdir(parents=True, exist_ok=False)
    interface_id = interface_identity()
    write_new(directory / "input.json", raw)
    write_new(directory / "resolved.json", spec)
    write_new(
        directory / "job.json",
        dict(
            schema="pas.job.v1",
            interface_id=interface_id,
            resolved_sha256=sha(directory / "resolved.json"),
            input_sha256=sha(directory / "input.json"),
        ),
    )
    with run_lock(directory), (directory / "run.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                print("PAS: preparing new run", flush=True)
                files, config = spec["files"], spec["config"]
                verify_resolved(spec)
                if "seed_file" in files:
                    if spec["initialization_mode"] == "frozen_paper_seed":
                        seed = read_json(files["seed_file"]["path"])
                        require(
                            identity(seed) == spec["paper_seed_id"], "Frozen seed identity changed"
                        )
                        write_new(directory / "initial/seed.json", seed)
                    else:
                        options = {"allow_compute": False}
                        if "role_reference_file" in files:
                            options["role_reference"] = read_json(
                                files["role_reference_file"]["path"]
                            )
                        seed = backend.import_seed(
                            files["seed_file"]["path"], config, directory / "initial", **options
                        )
                else:
                    from pas.cold_start import prepare_cold_start

                    scf = files.get("scf_source_file", {}).get("path")
                    seed = prepare_cold_start(
                        read_json(files["case_file"]["path"]), directory / "initial", config, scf
                    )
                options = dict(spec["policies"])
                if "bootstrap_evidence_file" in files:
                    options["bootstrap_input_evidence"] = files["bootstrap_evidence_file"]["path"]
                references = files.get("references_file", {}).get("path")
                controller.prepare(seed, config, directory / "engine", references, **options)
                if not prepare_only:
                    print("PAS: starting native search", flush=True)
                    controller.run(directory / "engine")
                result = _export(directory, spec, interface_id)
                print("PAS: " + result["status"], flush=True)
                return result
            except Exception as error:
                traceback.print_exc(file=log)
                _failure(directory, spec, error)
                raise


def _load_job(directory):
    job = read_json(directory / "job.json")
    require(job.get("schema") == "pas.job.v1", "Unsupported run directory")
    require(
        job["interface_id"] == interface_identity(),
        "Interface version changed; use the original installed version",
    )
    require(sha(directory / "resolved.json") == job["resolved_sha256"], "Resolved input changed")
    require(sha(directory / "input.json") == job["input_sha256"], "Stored input changed")
    return job, verify_resolved(read_json(directory / "resolved.json"))


def status(directory):
    verify_source()
    directory = Path(directory).resolve()
    job, spec = _load_job(directory)
    if not (directory / "engine/state.json").is_file():
        return read_json(directory / "result.json")
    _, state = engine.verify_run(directory / "engine")
    return state_result(state, spec, job["interface_id"], directory / "engine")[0]


def resume(directory):
    verify_source()
    directory = Path(directory).resolve()
    job, spec = _load_job(directory)
    require(
        (directory / "engine/state.json").is_file(),
        "Initialization did not complete; inspect run.log before preparing a new run",
    )
    require_cluster(directory, spec["config"])
    with run_lock(directory), (directory / "run.log").open("a", encoding="utf-8") as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                print("PAS: resuming the existing immutable ledger", flush=True)
                controller.run(directory / "engine")
                return _export(directory, spec, job["interface_id"])
            except Exception as error:
                traceback.print_exc(file=log)
                _failure(directory, spec, error)
                raise
