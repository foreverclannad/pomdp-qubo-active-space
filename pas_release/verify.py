"""Verify this release's engine, frozen reference data and optional evidence."""

from .common import (
    COMMIT,
    SOURCE,
    Evidence,
    interface_identity,
    read_json,
    require,
    sha,
    verify_source,
)
from .resources import DATA, PACKAGE


def validate_assets():
    frozen = read_json(DATA / "baseline.json")["reference_files"]
    for name, digest in frozen.items():
        require(sha(DATA / name) == digest, "Frozen reference material changed: " + name)
    return len(frozen)


def run(evidence=None):
    source = verify_source()
    references = validate_assets()
    count = Evidence(evidence).verify_all() if evidence else None
    return dict(
        passed=True,
        scientific_source_id=source,
        scientific_source_files=len(read_json(PACKAGE / "core_manifest.json")["files"]),
        numerical_baseline_commit=COMMIT,
        numerical_baseline_source_id=SOURCE,
        interface_id=interface_identity(),
        reference_files=references,
        evidence_files=count,
    )
