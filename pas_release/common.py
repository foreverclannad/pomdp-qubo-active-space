"""Strict I/O, identities and immutable evidence access for the public interface."""

import hashlib
import json
import math
from pathlib import Path

from .resources import DATA, PACKAGE

COMMIT = "c06614a37f5d9e0515b576b06999f1c39c229774"
SOURCE = "d78c57e6d16460b07643be6b02451ffeb46feba625ebce3cee04c3a139cd5a54"


class ValidationError(ValueError):
    """An input or scientific acceptance contract was not satisfied."""


def require(condition, message):
    if not condition:
        raise ValidationError(str(message))


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValidationError(f"Non-finite JSON value: {value}")


def read_json(path, *, expected_sha256=None):
    content = Path(path).read_bytes()
    if expected_sha256 is not None:
        require(
            hashlib.sha256(content).hexdigest() == expected_sha256,
            "Run input file changed: " + str(path),
        )
    try:
        return json.loads(
            content.decode("utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except json.JSONDecodeError as error:
        raise ValidationError(f"{path}:{error.lineno}:{error.colno}: {error.msg}") from error


def write_new(path, value):
    path = Path(path)
    # Serialize before opening the destination so invalid values leave no partial file.
    text = json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def compare_values(actual, expected, tolerance=1e-12, location="result"):
    if isinstance(expected, dict):
        require(
            isinstance(actual, dict) and actual.keys() == expected.keys(),
            location + ": fields differ",
        )
        return max(
            (
                compare_values(actual[k], v, tolerance, location + "." + k)
                for k, v in expected.items()
            ),
            default=0.0,
        )
    if isinstance(expected, list):
        require(
            isinstance(actual, list) and len(actual) == len(expected), location + ": length differs"
        )
        return max(
            (
                compare_values(a, b, tolerance, f"{location}[{i}]")
                for i, (a, b) in enumerate(zip(actual, expected))
            ),
            default=0.0,
        )
    if isinstance(expected, float):
        require(
            type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected),
            location + ": expected finite number",
        )
        error = abs(actual - expected)
        require(error <= tolerance, f"{location}: error {error:.6g} exceeds {tolerance:.6g}")
        return error
    require(type(actual) is type(expected) and actual == expected, location + ": value differs")
    return 0.0


def verify_source():
    from pas import controller
    from pas.common import identity

    require(__debug__, "Scientific validation requires Python without -O")
    baseline = read_json(DATA / "baseline.json")
    require(
        baseline["commit"] == COMMIT and baseline["source_id"] == SOURCE,
        "Numerical baseline identity changed",
    )
    manifest = read_json(PACKAGE / "core_manifest.json")
    require(
        manifest["schema"] == "pas.scientific-source.v1"
        and manifest["numerical_baseline"] == dict(commit=COMMIT, source_id=SOURCE),
        "Release source manifest has a different numerical baseline",
    )
    actual = controller.source_files()
    require(actual == manifest["files"], "Scientific files differ from the release manifest")
    source = identity(actual)
    require(source == manifest["source_id"], "Scientific source identity differs")
    return source


def interface_identity():
    from pas.common import identity

    files = {
        p.relative_to(PACKAGE).as_posix(): sha(p)
        for p in sorted(PACKAGE.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".json")
    }
    return identity(files)


class Evidence:
    """Original bytes are authenticated before creating relocated planning views."""

    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.tree = self.root / "tree"
        self.index = read_json(DATA / "evidence_manifest.json")["files"]
        self.used = {}

    def canonical(self, path):
        value = Path(path)
        if value.is_absolute() and value.is_relative_to(self.tree):
            return "/" + value.relative_to(self.tree).as_posix()
        return str(path)

    def local(self, path):
        name = self.canonical(path)
        require(name in self.index, "File is not in the evidence manifest: " + name)
        value = (self.root / self.index[name]["path"]).resolve()
        require(value.is_relative_to(self.tree), "Evidence path escapes its tree")
        return value

    def bind(self, path, expected=None):
        name = self.canonical(path)
        value = self.local(path)
        digest = sha(value)
        require(
            digest == self.index[name]["sha256"] and (expected is None or digest == expected),
            "Evidence hash mismatch: " + name,
        )
        require(value.stat().st_size == self.index[name]["size"], "Evidence size mismatch: " + name)
        self.used[str(value)] = digest
        return value

    def read(self, path, expected=None):
        return read_json(self.bind(path, expected))

    def artifact(self, record):
        return self.bind(record["path"], record["sha256"])

    def relocate(self, value):
        if isinstance(value, dict):
            return {k: self.relocate(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.relocate(v) for v in value]
        if isinstance(value, str) and value in self.index:
            return str(self.bind(value))
        return value

    def context(self, raw):
        from pas.contracts import Context

        original = Context.from_dict(raw)
        result = Context.from_dict(self.relocate(raw))
        require(result.context_id == original.context_id, "Relocation changed the context identity")
        return result

    def receipt(self, raw):
        from . import engine

        engine.verify_receipt(raw, lambda p: sha(self.bind(p)))

    def request(self, directory, context, receipt, config, *, fixed_dmrg_policy=None):
        from pas import backend
        from pas.common import identity
        from pas.diagnostic_window import survey_partition

        path = Path(directory) / "request.json"
        request = self.read(path)
        require(
            request["request_id"]
            == identity({k: v for k, v in request.items() if k != "request_id"}),
            "Request content identity changed",
        )
        require(
            request["request_id"] == receipt["request_id"], "Request and receipt identities differ"
        )
        part = (
            survey_partition(context, receipt["active"])
            if receipt["kind"] == "survey"
            else context.partition(receipt["active"])
        )
        expected = dict(
            schema="pas_physical_request_v1",
            context_id=context.context_id,
            basis_id=context.basis_id,
            checkpoint_hash=context.payload.get("checkpoint_hash", context.basis_id),
            kind=receipt["kind"],
            partition=part,
            protocol=config,
        )
        require(
            all(identity(request[k]) == identity(v) for k, v in expected.items()),
            "Request differs from the physical context/protocol",
        )
        for record in (request, receipt):
            require(
                record.get("fixed_dmrg_policy") == fixed_dmrg_policy, "Fixed DMRG policy changed"
            )
            if fixed_dmrg_policy is None:
                require(
                    "fixed_dmrg_policy" not in record,
                    "Legacy request cannot claim a new DMRG policy",
                )
        require(
            isinstance(request["input"], str) and request["input"].strip(), "Missing solver input"
        )
        require(
            self.artifact(receipt["artifacts"]["request"]) == self.local(path),
            "Wrong request artifact",
        )
        if receipt["kind"] == "survey" and context.backend == "cas":
            backend.verify_cas_survey(
                context, part, config, self.relocate(request), self.relocate(receipt)
            )
        return request

    def optimized(self, context, raw):
        from . import engine

        return engine.optimized(context, self.relocate(raw))

    def verify_all(self):
        for name in self.index:
            self.bind(name)
        return len(self.index)

    def unchanged(self):
        require(
            all(sha(path) == digest for path, digest in self.used.items()),
            "Evidence changed during replay",
        )
        return True
