"""CLI behavior and result exports without a native molecular calculation."""

import csv
import json
import subprocess
import sys

import pytest

from pas.common import file_hash, write_json
from pas_release import cli, outputs, runtime
from pas_release.common import ValidationError, read_json
from pas_release.profiles import PAPER_INPUTS
from pas_release.resources import CASE_NAMES, DATA


def test_module_entry_uses_the_public_command():
    run = subprocess.run(
        [sys.executable, "-m", "pas", "init", "--list", "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert set(json.loads(run.stdout)["templates"]) == set(CASE_NAMES) | {"cas", "dmrg"}


def test_generated_molecular_input_can_be_checked_and_never_overwrites(tmp_path, capsys):
    destination = tmp_path / "new calculation/input.json"
    assert cli.main(["init", "cas", "--output", str(destination), "--json"]) == 0
    generated = json.loads(capsys.readouterr().out)
    assert len(generated["created_files"]) == 2 and generated["validated"] is False
    assert read_json(destination)["profile"] == "cas"
    before = {path: path.read_bytes() for path in destination.parent.iterdir()}
    assert cli.main(["check", str(destination)]) == 0
    check = capsys.readouterr().out
    assert "2 electrons / 2 orbitals" in check and "Fixed random seed: 20261004" in check
    assert "Evidence (relative weights): occupation=1, composition=1" in check
    assert "Evidence workflow: posterior_evidence_capture_v1" in check
    assert "Composition alpha: 0.45" in check
    assert cli.main(["init", "cas", "--output", str(destination)]) == 2
    assert "already exists" in capsys.readouterr().err
    assert {path: path.read_bytes() for path in destination.parent.iterdir()} == before
    destination.unlink()
    assert cli.main(["init", "dmrg", "--output", str(destination)]) == 2
    assert "--case-file" in capsys.readouterr().err
    assert not destination.exists()  # An existing molecular file is protected too.


def test_templates_are_discoverable_and_bind_the_requested_sources(
    tmp_path, input_file, capsys, monkeypatch
):
    assert cli.main(["init", "--list", "--json"]) == 0
    assert set(json.loads(capsys.readouterr().out)["templates"]) == set(CASE_NAMES) | {
        "cas",
        "dmrg",
    }
    cases = read_json(DATA / "cases.json")["cases"]
    for profile, key in CASE_NAMES.items():
        destination = tmp_path / (profile + ".json")
        assert cli.main(["init", profile, "--output", str(destination)]) == 0
        value = read_json(destination)
        assert value["initialization"] == {"seed_file": cases[key]["original_run"] + "/seed.json"}
        assert value["evidence"] == dict(
            enabled=cases[key]["config"]["channels"],
            weights=cases[key]["config"]["weights"],
            composition_alpha=0.45,
        )
        assert {field: value[field] for field in PAPER_INPUTS[key]} == {
            field: binding["path"] for field, binding in PAPER_INPUTS[key].items()
        }
        assert "system" not in value and "settings" not in value
    monkeypatch.chdir(input_file.parent)
    destination = input_file.parent / "nested/input.json"
    assert (
        cli.main(["init", "cas", "--case-file", "molecule.json", "--output", str(destination)]) == 0
    )
    assert read_json(destination)["initialization"] == {"case_file": "../molecule.json"}
    capsys.readouterr()
    assert cli.main(["check", str(destination), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["input_valid"] is True
    destination = input_file.parent / "custom.json"
    assert (
        cli.main(
            ["init", "dio-s1min-dmrg", "--case-file", "molecule.json", "--output", str(destination)]
        )
        == 0
    )
    assert "references_file" not in read_json(destination)
    assert "bootstrap_evidence_file" not in read_json(destination)


def test_check_writes_single_json_and_does_not_create_outputs(input_file, capsys, monkeypatch):
    monkeypatch.setattr(
        runtime.backend, "execute", lambda *a, **k: pytest.fail("check must not compute")
    )
    before = sorted(input_file.parent.rglob("*"))
    assert cli.main(["check", str(input_file), "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["input_valid"] is True and result["system"]["orbital_index_base"] == 1
    assert sorted(input_file.parent.rglob("*")) == before


def test_input_error_is_nonzero_and_only_on_stderr(tmp_path, capsys):
    assert cli.main(["check", str(tmp_path / "missing.json"), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["exit_code"] == 2


def test_native_run_on_wrong_host_fails_before_creating_directory(input_file, monkeypatch):
    from pas_release import cluster

    monkeypatch.setattr(cluster.socket, "gethostname", lambda: "not-cu07")
    destination = input_file.parent / "new-run"
    assert cli.main(["run", str(input_file), "--output", str(destination)]) == 3
    assert not destination.exists()


@pytest.mark.parametrize(
    "profile, enabled, reason",
    [
        ("cas", ["occupation", "composition", "qit"], "CAS backend"),
        ("cas", ["occupation"], "posterior_evidence_capture_v1"),
        ("dmrg", ["occupation", "qit"], "posterior_evidence_capture_v1"),
    ],
)
def test_incompatible_evidence_never_switches_workflow_or_starts_run(
    input_file, monkeypatch, capsys, profile, enabled, reason
):
    value = read_json(input_file)
    value.update(profile=profile, evidence=dict(enabled=enabled))
    input_file.write_text(json.dumps(value))
    monkeypatch.setattr(
        runtime,
        "require_cluster",
        lambda *a: pytest.fail("invalid input reached execution preflight"),
    )
    destination = input_file.parent / "must-not-exist"
    assert cli.main(["run", str(input_file), "--output", str(destination), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    message = json.loads(captured.err)["error"]
    assert "evidence.enabled" in message and reason in message
    assert not destination.exists()
    assert read_json(input_file) == value


def test_existing_output_cannot_be_overwritten(input_file):
    with pytest.raises(ValidationError, match="already exists"):
        runtime.start(input_file, input_file.parent)


@pytest.mark.parametrize("changed_at", ["validation", "preflight"])
def test_input_edit_during_start_is_rejected_before_creating_run(
    input_file, monkeypatch, changed_at
):
    from pas import cold_start

    def edit_input(*args):
        value = read_json(input_file)
        value["settings"] = {"threads": 1}
        input_file.write_text(json.dumps(value))

    monkeypatch.setattr(runtime, "require_cluster", lambda *a: None)
    if changed_at == "validation":
        validate = runtime.engine.validate_policies

        def validate_then_edit(*args):
            validate(*args)
            edit_input()

        monkeypatch.setattr(runtime.engine, "validate_policies", validate_then_edit)
    else:
        monkeypatch.setattr(runtime, "require_cluster", edit_input)
    monkeypatch.setattr(
        cold_start, "prepare_cold_start", lambda *a: pytest.fail("changed input reached the solver")
    )
    destination = input_file.parent / "changed-run"
    with pytest.raises(ValidationError, match="input file changed"):
        runtime.start(input_file, destination)
    assert not destination.exists()


def test_public_output_is_one_based_full_precision_and_has_units(tmp_path):
    context = dict(
        backend="dmrg", electrons=2, active=[0, 2], states=dict(roots=[1, 2], weights=[0.5, 0.5])
    )
    receipt = dict(
        kind="optimize",
        active=[1, 3],
        valid=True,
        converged=True,
        state_valid=True,
        metrics=dict(quality=0.4567890123456789, energy=-12.345678901234567),
        output_context=context,
        cpu_seconds=2.5,
        wall_seconds=1.5,
        context_id="context",
    )
    path = tmp_path / "receipt.json"
    write_json(path, receipt)
    state = dict(
        context=context,
        incumbent=receipt,
        status="budget_exhausted",
        events=[dict(number=1, receipt="receipt.json", sha256=file_hash(path), accepted=True)],
        remaining_diagnostics=1,
        remaining_optimizations=0,
        search_cpu_seconds=2.5,
        search_wall_seconds=1.5,
        pending=None,
    )
    spec = dict(
        name="fixture",
        profile="tif3",
        config_id="config",
        historical_profile_unchanged=True,
        paper_profile_unchanged=True,
        config=dict(composition_purity_weight=0.45),
    )
    result, rows = outputs.state_result(state, spec, "interface", tmp_path)
    outputs.write_results(tmp_path, result, rows)
    saved = read_json(tmp_path / "result.json")
    assert saved["metrics"] == receipt["metrics"]
    assert saved["active_space"]["indices"] == [1, 3]
    assert saved["units"]["energy"] == "hartree"
    assert saved["root_energies_hartree"] is None  # No guessed DMRG container endpoint.
    with (tmp_path / "history.csv").open() as stream:
        history = list(csv.DictReader(stream))
    assert history[0]["input_active_indices"] == "2 4"
    assert float(history[0]["energy_hartree"]) == receipt["metrics"]["energy"]
    assert float(history[0]["composition_alpha"]) == 0.45
    assert history[0]["historical_quality"] == ""


def test_mps_root_energy_endpoint_is_used_and_tampering_is_rejected(tmp_path):
    path = tmp_path / "mps.json"
    write_json(path, dict(root_energies=[-1.2, -1.0]))
    receipt = dict(artifacts=dict(mps_evidence=dict(path=str(path), sha256=file_hash(path))))
    assert outputs._root_energies(receipt) == [-1.2, -1.0]
    path.write_text("{}")
    with pytest.raises(ValidationError, match="changed"):
        outputs._root_energies(receipt)


def test_failed_initialization_leaves_an_inspectable_result(input_file, monkeypatch):
    from pas import cold_start

    monkeypatch.setattr(runtime, "require_cluster", lambda *a: {})

    def fail(*args):
        raise RuntimeError("solver initialization failed")

    monkeypatch.setattr(cold_start, "prepare_cold_start", fail)
    destination = input_file.parent / "failed-run"
    with pytest.raises(RuntimeError, match="solver initialization failed"):
        runtime.start(input_file, destination)
    assert read_json(destination / "result.json")["status"] == "failed"
    assert "solver initialization failed" in (destination / "run.log").read_text()
    assert runtime.status(destination)["status"] == "failed"


def test_resume_rejects_changed_input_before_engine_call(input_file, monkeypatch):
    from pas import cold_start

    monkeypatch.setattr(runtime, "require_cluster", lambda *a: {})
    monkeypatch.setattr(
        cold_start, "prepare_cold_start", lambda *a: (_ for _ in ()).throw(RuntimeError("stop"))
    )
    destination = input_file.parent / "run"
    with pytest.raises(RuntimeError):
        runtime.start(input_file, destination)
    (destination / "input.json").write_text("{}")
    monkeypatch.setattr(
        runtime.controller, "run", lambda *a: pytest.fail("changed run must not execute")
    )
    with pytest.raises(ValidationError, match="Stored input changed"):
        runtime.resume(destination)


def test_reproduce_requires_evidence_and_never_reuses_output(tmp_path, capsys):
    destination = tmp_path / "result"
    assert cli.main(["reproduce", "paper", "--output", str(destination)]) == 2
    assert not destination.exists()
    destination.mkdir()
    assert cli.main(["reproduce", "small-molecules", "--output", str(destination)]) == 2
