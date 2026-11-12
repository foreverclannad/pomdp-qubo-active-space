"""Build the curated source archive and wheel; no publication or solver execution."""

import argparse
import gzip
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory

from . import RELEASE
from .common import COMMIT, read_json, require, sha, write_new
from .resources import DATA, PACKAGE
from .verify import run as verify

LABEL = "pas-" + RELEASE
DIRECTORIES = ("pas", "pas_release", "examples", "docs", "tests", "requirements", "provenance")
# Development inventories stay in Git, outside the reader's distribution.
EXCLUDED = (
    "docs/maintenance/",
    "docs/paper/",
    "examples/native/",
    "examples/cold-start/",
    "provenance/audit-c06614a/",
    "provenance/licenses/",
)
ROOT_FILES = (
    "README.md",
    "CITATION.cff",
    "LICENSE",
    "NOTICE",
    "pyproject.toml",
    ".gitattributes",
    ".gitignore",
)


def files_for_release(source):
    paths = [source / name for name in ROOT_FILES]
    for directory in DIRECTORIES:
        paths.extend(
            path
            for path in (source / directory).rglob("*")
            if path.is_file()
            and not path.relative_to(source).as_posix().startswith(EXCLUDED)
            and not any(part in ("__pycache__", ".pytest_cache") for part in path.parts)
            and path.suffix not in (".pyc", ".pyo")
        )
    for path in paths:
        require(
            path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(source),
            "Invalid release path: " + str(path),
        )
    return sorted(paths)


def write_archive(destination, members):
    with destination.open("xb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
                for name, content in members:
                    info = tarfile.TarInfo(name)
                    info.mode, info.mtime = 0o644, 0
                    if isinstance(content, Path):
                        info.size = content.stat().st_size
                        with content.open("rb") as stream:
                            archive.addfile(info, stream)
                    else:
                        info.size = len(content)
                        archive.addfile(info, io.BytesIO(content))


def build(source, destination, evidence=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    require(source.is_dir(), "Source directory does not exist")
    require(
        source / "pas_release" == PACKAGE,
        "--source must match the source tree running this packager; run it from that checkout",
    )
    require(not destination.exists(), "Choose a new distribution directory")
    if evidence is not None:
        require(
            not destination.is_relative_to(Path(evidence).resolve()),
            "Output must be outside evidence",
        )
    verified = verify(evidence)
    paths = files_for_release(source)
    manifest = {path.relative_to(source).as_posix(): sha(path) for path in paths}
    destination.mkdir(parents=True, exist_ok=False)
    archive = destination / (LABEL + "-source.tar.gz")
    content = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    members = [(LABEL + "/" + path.relative_to(source).as_posix(), path) for path in paths]
    write_archive(archive, members + [(LABEL + "/release_files.json", content)])
    with TemporaryDirectory(prefix="pas-package-") as temporary:
        # Build from exactly the files in the archive. Setuptools may otherwise
        # carry removed modules from an earlier source/build/lib into the wheel.
        build_tree = Path(temporary)
        with tarfile.open(archive) as packaged:
            for name in manifest:
                path = build_tree / name
                path.parent.mkdir(parents=True, exist_ok=True)
                with packaged.extractfile(LABEL + "/" + name) as stream:
                    path.write_bytes(stream.read())
        _build_wheel(build_tree, destination)
    require(
        all(sha(source / name) == digest for name, digest in manifest.items()),
        "Source changed during packaging",
    )
    wheel = list(destination.glob("*.whl"))
    require(len(wheel) == 1, "Expected exactly one wheel")
    artifacts = {
        path.name: dict(sha256=sha(path), bytes=path.stat().st_size) for path in (archive, *wheel)
    }
    report = dict(
        release=LABEL,
        numerical_baseline=COMMIT,
        verification=verified,
        source_files=len(manifest) + 1,
        artifacts=artifacts,
        evidence=dict(
            manifest_sha256=sha(DATA / "evidence_manifest.json"),
            files=len(read_json(DATA / "evidence_manifest.json")["files"]),
        ),
    )
    write_new(destination / "package_manifest.json", report)
    (destination / "SHA256SUMS").write_text(
        "".join(value["sha256"] + "  " + name + "\n" for name, value in artifacts.items()),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    return report


def _build_wheel(source, destination):
    with (destination / "build.log").open("x", encoding="utf-8") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--disable-pip-version-check",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(destination),
                str(source),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evidence", type=Path)
    args = parser.parse_args()
    build(args.source, args.output, args.evidence)


if __name__ == "__main__":
    main()
