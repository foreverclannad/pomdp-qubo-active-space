# Selected from c06614a test_work_storage.py; retention rationale: docs/maintenance/retention.json
"""Execution storage is configurable without changing physical protocols."""

import os
import sys

import pytest

from pas import backend


@pytest.fixture
def storage(tmp_path, monkeypatch):
    root = tmp_path / "work"
    root.mkdir()
    monkeypatch.setenv("PAS_WORK_ROOT", str(root))
    monkeypatch.setattr(backend.socket, "gethostname", lambda: "cu07")
    monkeypatch.setattr(os, "getuid", lambda: root.stat().st_uid, raising=False)
    return root


def test_configured_owned_work_root(storage):
    actual = backend.checked_directory(storage / "action", 16)
    assert actual == (storage / "action").resolve()
    assert actual.is_dir()


@pytest.mark.skipif(sys.platform == "win32", reason="Native Linux symlinks")
def test_symlink_cannot_escape_work_root(storage):
    outside = storage.parent / "external"
    outside.mkdir()
    (storage / "link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="work root"):
        backend.checked_directory(storage / "link/action", 16)
    assert not (outside / "action").exists()


def test_root_requires_ownership(storage, monkeypatch):
    monkeypatch.setattr(os, "getuid", lambda: storage.stat().st_uid + 1)
    with pytest.raises(RuntimeError, match="owned"):
        backend.checked_directory(storage / "action", 16)


def test_host_gate_is_preserved(storage, monkeypatch):
    monkeypatch.setattr(backend.socket, "gethostname", lambda: "other-node")
    with pytest.raises(RuntimeError, match="cu07"):
        backend.checked_directory(storage / "action", 16)


@pytest.mark.parametrize("threads", [0, 33])
def test_cpu_limit_is_preserved(storage, threads):
    with pytest.raises(backend.ContractError, match="CPU limit"):
        backend.checked_directory(storage / "action", threads)
