"""Release builds must use the verified source without stale build products."""

import tarfile
from pathlib import Path

import pytest

from pas_release import package
from pas_release.common import ValidationError


def test_packaging_rejects_a_different_source_before_verification(tmp_path, monkeypatch):
    monkeypatch.setattr(package, "verify", lambda *a: pytest.fail("verified the wrong source"))
    destination = tmp_path / "dist"
    with pytest.raises(ValidationError, match="--source must match"):
        package.build(tmp_path, destination)
    assert not destination.exists()


def test_source_archive_and_wheel_share_a_clean_build_tree(tmp_path, monkeypatch):
    source, destination = tmp_path / "source", tmp_path / "dist"
    module = source / "pas_release/current.py"
    module.parent.mkdir(parents=True)
    module.write_text("CURRENT = True\n")
    (source / "pyproject.toml").write_text("# fixture\n")
    stale = source / "build/lib/pas_release/removed.py"
    stale.parent.mkdir(parents=True)
    stale.write_text("# stale module left by a previous wheel build\n")
    monkeypatch.setattr(package, "PACKAGE", module.parent, raising=False)
    monkeypatch.setattr(package, "ROOT_FILES", ("pyproject.toml",))
    monkeypatch.setattr(package, "DIRECTORIES", ("pas_release",))
    monkeypatch.setattr(package, "verify", lambda *a: {"passed": True})

    def build_wheel(command, **options):
        tree = Path(command[-1])
        assert tree != source and not (tree / "build").exists()
        assert {p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file()} == {
            "pyproject.toml",
            "pas_release/current.py",
        }
        assert (tree / "pas_release/current.py").read_bytes() == module.read_bytes()
        (destination / "fixture.whl").write_bytes(b"wheel fixture")

    monkeypatch.setattr(package.subprocess, "run", build_wheel)
    report = package.build(source, destination)
    with tarfile.open(destination / (package.LABEL + "-source.tar.gz")) as archive:
        assert len(archive.getnames()) == report["source_files"] == 3
        assert not any("removed.py" in name for name in archive.getnames())
    assert stale.is_file()  # Packaging does not clean or modify the source tree.
