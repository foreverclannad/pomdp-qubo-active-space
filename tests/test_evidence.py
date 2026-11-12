"""Failure-path coverage for release adapters, independent of scientific tests."""

import hashlib
import json

import pytest

from pas_release import common, engine


@pytest.fixture
def archive(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "paper").mkdir(parents=True)
    bundle = tmp_path / "evidence"
    (bundle / "tree/run").mkdir(parents=True)
    content = b"original immutable observation"
    target = bundle / "tree/run/data"
    target.write_bytes(content)
    index = {
        "/run/data": dict(
            path="tree/run/data", sha256=hashlib.sha256(content).hexdigest(), size=len(content)
        )
    }
    (source / "paper/evidence_manifest.json").write_text(json.dumps(dict(files=index)))
    monkeypatch.setattr(common, "DATA", source / "paper")
    return common.Evidence(bundle), target


def test_changed_evidence_is_rejected(archive):
    evidence, target = archive
    assert evidence.verify_all() == 1
    target.write_bytes(b"tampered immutable observation")
    with pytest.raises(ValueError, match="hash mismatch"):
        evidence.bind("/run/data")


def test_release_source_is_distinct_from_baseline_and_rejects_tampering(tmp_path, monkeypatch):
    from pas import controller

    assert common.verify_source() == controller.source_identity() != common.SOURCE
    manifest = common.read_json(common.PACKAGE / "core_manifest.json")
    path = tmp_path / "core_manifest.json"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(common, "PACKAGE", tmp_path)
    files = controller.source_files()
    monkeypatch.setattr(controller, "source_files", lambda: {**files, "unreviewed.py": "changed"})
    with pytest.raises(common.ValidationError, match="release manifest"):
        common.verify_source()
    monkeypatch.setattr(controller, "source_files", lambda: files)
    manifest["source_id"] = common.SOURCE
    path.write_text(json.dumps(manifest))
    with pytest.raises(common.ValidationError, match="source identity differs"):
        common.verify_source()


def test_changed_during_validation_is_rejected(archive):
    evidence, target = archive
    evidence.bind("/run/data")
    target.write_bytes(b"changed after authentication")
    with pytest.raises(ValueError, match="changed during"):
        evidence.unchanged()


def test_unknown_path_never_falls_back_to_original_machine(archive):
    evidence, _ = archive
    with pytest.raises(ValueError, match="not in the evidence manifest"):
        evidence.bind("/etc/passwd")


def test_manifest_path_cannot_escape_bundle(archive):
    evidence, _ = archive
    evidence.index["/run/data"]["path"] = "../../outside"
    with pytest.raises(ValueError, match="escapes"):
        evidence.local("/run/data")


def test_relocation_does_not_mutate_raw_record(archive):
    evidence, target = archive
    original = dict(
        artifacts=dict(data=dict(path="/run/data", sha256=evidence.index["/run/data"]["sha256"]))
    )
    copied = evidence.relocate(original)
    assert copied["artifacts"]["data"]["path"] == str(target)
    assert original["artifacts"]["data"]["path"] == "/run/data"


def test_numerical_acceptance_rejects_discrete_or_material_changes():
    common.compare_values(
        dict(energy=-1.0, active=[1, 2]), dict(energy=-1.0 + 1e-14, active=[1, 2])
    )
    with pytest.raises(common.ValidationError):
        common.compare_values(dict(active=[2, 1]), dict(active=[1, 2]))
    with pytest.raises(common.ValidationError):
        common.compare_values(1.0, 1.0 + 1e-8)
    with pytest.raises(common.ValidationError):
        common.compare_values(float("nan"), 1.0)


def test_read_only_mode_blocks_solver_and_ledger():
    from pas import backend, controller

    with engine.read_only():
        with pytest.raises(RuntimeError, match="prohibited"):
            backend.execute(None)
        with pytest.raises(RuntimeError, match="prohibited"):
            controller.write_json(None, None)


def test_ci_cc_uses_stable_singular_values_and_rejects_subspace_change():
    from copy import deepcopy

    from pas_release.reproduction.replay import check_ci_cc

    expected = common.read_json(common.DATA / "expected_results.json")["results"]["CI_CC"]
    changed = deepcopy(expected)
    changed["comparison"]["principal_angles_radians"][0] += 1e-9
    assert check_ci_cc(changed, expected) == 0
    changed["comparison"]["singular_values"][0] -= 1e-8
    with pytest.raises(common.ValidationError):
        check_ci_cc(changed, expected)
