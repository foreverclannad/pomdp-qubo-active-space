"""Versioned public JSON inputs, resolved without launching a solver."""

import math
import re
from copy import deepcopy
from difflib import get_close_matches
from pathlib import Path

from pas.common import identity
from pas.config import CHANNELS, DEFAULTS, configuration
from pas.contracts import Context

from . import engine
from .common import ValidationError, read_json, require, sha
from .profiles import check_paper_materials, paper_configuration
from .resources import CASE_NAMES, DATA, PROFILE_NAMES

SCHEMA = "pas.input.v1"
SETTINGS = {"threads", "diagnostic_budget", "optimization_budget", "seed"}
SYSTEM_FIELDS = {
    "active_electrons",
    "active_orbitals",
    "spin_multiplicity",
    "roots",
    "weights",
    "active_indices",
}


def object_fields(value, allowed, location):
    require(
        isinstance(value, dict),
        location + ': expected a JSON object, for example {"field": "value"}',
    )
    unknown = sorted(set(value) - allowed)
    if unknown:
        field = unknown[0]
        match = get_close_matches(field, sorted(allowed), n=1, cutoff=0.65)
        hint = (
            f"Use '{match[0]}' instead."
            if match
            else "Allowed fields: " + ", ".join(sorted(allowed))
        )
        raise ValidationError(f"{location}.{field}: unknown field. {hint}")


def required_fields(value, fields, location):
    missing = sorted(set(fields) - value.keys())
    require(not missing, f"{location}: missing required fields: {', '.join(missing)}")


def evidence_options(value, baseline, profile):
    """Translate explicit evidence choices without changing the profile's workflow."""
    object_fields(value, {"enabled", "weights", "composition_alpha"}, "evidence")
    if not value:
        return {}
    enabled = value.get("enabled", baseline["channels"])
    require(
        isinstance(enabled, list) and enabled and all(isinstance(c, str) for c in enabled),
        "evidence.enabled: provide a nonempty list of occupation, composition and/or qit",
    )
    require(
        len(set(enabled)) == len(enabled), "evidence.enabled: each channel may appear only once"
    )
    require(
        set(enabled) <= set(CHANNELS),
        "evidence.enabled: unknown channel; choose occupation, composition or qit",
    )
    enabled = [channel for channel in CHANNELS if channel in enabled]
    weights = value.get("weights", {c: baseline["weights"].get(c, 1.0) for c in enabled})
    object_fields(weights, set(CHANNELS), "evidence.weights")
    require(
        set(weights) == set(enabled),
        "evidence.weights: provide exactly one weight for each enabled channel: "
        + ", ".join(enabled)
        + "; omit weights for disabled channels",
    )
    converted = {}
    for channel in enabled:
        weight = weights[channel]
        message = f"evidence.weights.{channel}: expected a finite positive number; got {weight!r}"
        require(type(weight) in (int, float), message)
        try:
            number = float(weight)
        except (OverflowError, ValueError) as error:
            raise ValidationError(message) from error
        require(math.isfinite(number) and number > 0, message)
        converted[channel] = number
    require(
        math.isfinite(sum(converted.values())),
        "evidence.weights: the weight sum overflows; use smaller positive relative weights",
    )
    require(
        baseline["backend"] != "cas" or "qit" not in enabled,
        "evidence.enabled: the native CAS backend does not supply qit evidence; "
        "this profile requires occupation and composition",
    )
    if baseline.get("survey_model") == "posterior_evidence_capture_v1":
        require(
            set(enabled) == set(baseline["channels"]),
            f"evidence.enabled: profile '{profile}' uses posterior_evidence_capture_v1, "
            "which requires "
            + ", ".join(baseline["channels"])
            + "; requested "
            + ", ".join(enabled)
            + ". Keep the required channels; the algorithm workflow is not switched automatically",
        )
    alpha = value.get("composition_alpha", baseline["composition_purity_weight"])
    require(
        type(alpha) in (int, float) and 0 <= alpha <= 1,
        "evidence.composition_alpha: expected a finite number in [0, 1]; paper uses 0.45",
    )
    return dict(channels=enabled, weights=converted, composition_purity_weight=float(alpha))


def orbital_indices(values, count):
    require(
        isinstance(values, list) and all(type(i) is int and i >= 1 for i in values),
        "system.active_indices must contain one-based positive integers",
    )
    require(
        len(values) == count and len(set(values)) == count,
        "system.active_indices must contain exactly active_orbitals distinct indices",
    )
    return [value - 1 for value in values]


def check_states(electrons, orbitals, states):
    required_fields(states, {"roots", "weights", "spin_multiplicity"}, "states")
    require(
        type(electrons) is int
        and type(orbitals) is int
        and orbitals > 0
        and 0 <= electrons <= 2 * orbitals,
        f"active_space: electrons={electrons!r}, orbitals={orbitals!r}; use positive integer orbitals and 0 <= electrons <= 2 * orbitals",
    )

    roots, weights = states.get("roots"), states.get("weights")
    spin = states.get("spin_multiplicity")
    require(
        type(spin) is int
        and spin >= 1
        and spin - 1 <= min(electrons, 2 * orbitals - electrons)
        and (electrons - spin + 1) % 2 == 0,
        f"states.spin_multiplicity: {spin!r} is incompatible with {electrons} active electrons in {orbitals} orbitals; correct the electron count or multiplicity",
    )
    require(
        isinstance(roots, list)
        and roots
        and all(type(r) is int for r in roots)
        and roots == list(range(1, len(roots) + 1)),
        "states.roots: use consecutive integer root numbers starting at 1, for example [1, 2]",
    )
    require(
        isinstance(weights, list)
        and len(weights) == len(roots)
        and all(type(w) in (int, float) and math.isfinite(w) and w > 0 for w in weights)
        and math.isclose(sum(weights), 1.0, rel_tol=0, abs_tol=1e-12),
        f"states.weights: {weights!r}; provide one positive weight per root with a total of 1, for example [0.5, 0.5] for two roots",
    )


def check_geometry(text):
    require(isinstance(text, str), "case_file.geometry_xyz must be an XYZ string in angstrom")
    lines = text.strip().splitlines()
    require(
        len(lines) >= 3 and lines[0].strip().isdigit(),
        "XYZ requires atom count, comment, and coordinates",
    )
    count = int(lines[0].strip())
    atoms = [line.split() for line in lines[2:] if line.strip()]
    require(count > 0 and len(atoms) == count, "XYZ atom count disagrees with coordinates")
    for number, atom in enumerate(atoms, 1):
        require(
            len(atom) == 4 and re.fullmatch(r"[A-Z][a-z]?", atom[0]),
            f"Invalid XYZ atom row {number}",
        )
        try:
            coordinates = [float(value) for value in atom[1:]]
        except ValueError as error:
            raise ValidationError(f"Invalid XYZ coordinates at atom {number}") from error
        require(
            all(math.isfinite(value) for value in coordinates),
            f"Non-finite XYZ coordinates at atom {number}",
        )


def _path(value, base, label):
    require(isinstance(value, str) and value.strip(), label + " must be a nonempty file path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    require(
        path.is_file(),
        f"{label}: file not found: {path}. Correct the path or restore the file; "
        f"relative paths are resolved from {base}",
    )
    return {"path": str(path), "sha256": sha(path)}


def _check_bound_files(value):
    if isinstance(value, dict):
        if {"path", "sha256"} <= value.keys():
            path = Path(value["path"])
            require(path.is_absolute(), "Bound physical file paths must be absolute: " + str(path))
            require(
                path.is_file(),
                f"initialization.seed_file: required physical file is missing: {path}. "
                "Restore the original file on the cluster; editing the seed path cannot replace it",
            )
            require(sha(path) == value["sha256"], "Bound physical file changed: " + str(path))
        for item in value.values():
            _check_bound_files(item)
    elif isinstance(value, list):
        for item in value:
            _check_bound_files(item)


def _settings(settings):
    object_fields(settings, SETTINGS, "settings")
    for key, value in settings.items():
        require(
            type(value) is int,
            f"settings.{key}: expected an integer, got {value!r}; true/false are not integers",
        )
        if key == "threads":
            require(
                1 <= value <= 32,
                f"settings.threads: expected 1–32, got {value}; the original cluster limit is 32",
            )
        elif key.endswith("budget"):
            require(
                0 <= value <= 6,
                f"settings.{key}: expected 0–6, got {value}; the frozen production plan permits at most six",
            )
        else:
            require(
                value == DEFAULTS["seed"],
                f"settings.seed: fixed at {DEFAULTS['seed']} by the paper protocol; remove this field or use that value",
            )
    return settings


def _initial_files(raw, base):
    initial = raw.get("initialization")
    object_fields(
        initial,
        {"seed_file", "case_file", "scf_source_file", "role_reference_file"},
        "initialization",
    )
    require(
        ("seed_file" in initial) != ("case_file" in initial),
        "initialization: specify exactly one of seed_file (a complete PAS seed) or case_file (a molecular JSON file)",
    )
    require(
        "role_reference_file" not in initial or "seed_file" in initial,
        "role_reference_file is only supported with a seed_file",
    )
    require(
        "scf_source_file" not in initial or "case_file" in initial,
        "scf_source_file is only supported with a case_file",
    )
    files = {key: _path(value, base, "initialization." + key) for key, value in initial.items()}
    for key in ("references_file", "bootstrap_evidence_file"):
        if key in raw:
            files[key] = _path(raw[key], base, key)
    return files


def _load_seed(files, case, config, profile, verify_bindings):
    seed = read_json(files["seed_file"]["path"], expected_sha256=files["seed_file"]["sha256"])
    require(
        isinstance(seed, dict) and {"context", "incumbent", "initialization"} <= seed.keys(),
        "initialization.seed_file: expected a complete signed PAS initialization bundle; use case_file for a molecular definition",
    )
    seed_id = identity(seed)
    context = Context.from_dict(seed["context"])
    receipt = seed["incumbent"]
    require(
        isinstance(receipt, dict)
        and receipt.get("kind") == "initial"
        and all(receipt.get(k) for k in ("valid", "converged", "state_valid")),
        "seed_file must contain a converged initial result",
    )
    require(
        receipt.get("evidence_id")
        == identity({k: v for k, v in receipt.items() if k != "evidence_id"})
        and identity(receipt.get("output_context")) == identity(seed["context"]),
        "Initial receipt identity changed",
    )
    require(context.backend == config["backend"], "Seed backend differs from the selected profile")
    if profile in CASE_NAMES and seed_id == case["seed_id"]:
        check_paper_materials(PROFILE_NAMES[profile], files)
    # The five archived seeds have already been accepted with these exact
    # profiles. Reuse their signed context, including derived fixed-MO
    # containers, through controller.prepare's original seed contract.
    frozen_seed = (
        seed_id == case["seed_id"]
        and config == case["config"]
        and context.effective_config_id == identity(config)
        and "role_reference_file" not in files
    )
    initialization_mode = "frozen_paper_seed" if frozen_seed else "imported_seed"
    if verify_bindings:
        _check_bound_files(seed)
        require(
            sha(context.payload["basis_file"]) == context.basis_id, "Seed orbital basis changed"
        )
        require(
            sha(context.payload["checkpoint"]) == context.payload["checkpoint_hash"],
            "Seed checkpoint changed",
        )
        artifact = receipt.get("artifacts", {}).get("mps_evidence")
        if artifact and not frozen_seed:
            mps = read_json(artifact["path"])
            require(
                mps.get("orbital_basis_id") == context.basis_id,
                "This derived MPS seed requires its unchanged frozen paper profile; "
                "a changed configuration requires a compatible new initialization",
            )
    return context, initialization_mode


def _load_molecule(binding):
    molecule = read_json(binding["path"], expected_sha256=binding["sha256"])
    allowed = {
        "key",
        "case_id",
        "geometry_xyz",
        "basis",
        "basis_library",
        "basis_by_element",
        "cholesky",
        "charge",
        "relativistic",
        "group",
        "states",
        "active_space",
        "target",
    }
    object_fields(molecule, allowed, "case_file")
    required_fields(
        molecule,
        {"geometry_xyz", "basis", "charge", "relativistic", "states", "active_space"},
        "case_file",
    )
    object_fields(molecule["active_space"], {"electrons", "orbitals"}, "case_file.active_space")
    required_fields(molecule["active_space"], {"electrons", "orbitals"}, "case_file.active_space")
    object_fields(
        molecule["states"], {"roots", "weights", "spin_multiplicity", "ids"}, "case_file.states"
    )
    check_geometry(molecule["geometry_xyz"])
    require(isinstance(molecule["basis"], str) and molecule["basis"].strip(), "Missing basis")
    require(type(molecule["charge"]) is int, "Molecular charge must be an integer")
    return molecule


def load_input(path, *, verify_bindings=True):
    from .verify import validate_assets

    validate_assets()
    path = Path(path).resolve()
    require(
        path.is_file(),
        f"Input file not found: {path}. Generate a template with 'pas init --list' and 'pas init PROFILE --output input.json'",
    )
    input_digest = sha(path)
    raw = read_json(path, expected_sha256=input_digest)
    object_fields(
        raw,
        {
            "schema",
            "name",
            "profile",
            "initialization",
            "system",
            "settings",
            "evidence",
            "references_file",
            "bootstrap_evidence_file",
        },
        "input",
    )
    required_fields(raw, {"schema", "name", "profile", "initialization"}, "input")
    require(raw.get("schema") == SCHEMA, f"schema must be {SCHEMA}")
    require(
        isinstance(raw.get("name"), str)
        and bool(raw["name"].strip())
        and not any(ord(c) < 32 for c in raw["name"]),
        "name must be a nonempty single-line string",
    )
    require(
        isinstance(raw.get("profile"), str) and raw["profile"] in PROFILE_NAMES,
        "input.profile: unknown profile. Run 'pas init --list'; choose " + ", ".join(PROFILE_NAMES),
    )
    case = read_json(DATA / "cases.json")["cases"][PROFILE_NAMES[raw["profile"]]]
    settings = _settings(raw.get("settings", {}))
    paper_config = paper_configuration(case)
    options = {**paper_config, **settings}
    if "evidence" in raw:
        options.update(evidence_options(raw["evidence"], paper_config, raw["profile"]))
    config = configuration(options)
    policies = deepcopy(case["policies"])
    engine.validate_policies(config, policies)
    files = _initial_files(raw, path.parent)
    context = None
    initialization_mode = "cold_start"
    if "seed_file" in files:
        context, initialization_mode = _load_seed(
            files, case, config, raw["profile"], verify_bindings
        )
        electrons, orbitals, states = context.electrons, context.k, context.states
        active = list(context.active)
    else:
        molecule = _load_molecule(files["case_file"])
        electrons = molecule["active_space"]["electrons"]
        orbitals = molecule["active_space"]["orbitals"]
        states = molecule["states"]
        active = None
    check_states(electrons, orbitals, states)
    if context is None:
        require(
            electrons % 2 == 0 and states.get("spin_multiplicity") == 1,
            "Cold initialization supports closed-shell singlets; supply a verified seed for open-shell cases",
        )
    from pas import backend

    if context is not None:
        backend.check_capabilities(context, "optimize", config["channels"], config)
    else:
        backend.check_initial_capabilities(molecule, config)
    system = dict(
        active_electrons=electrons,
        active_orbitals=orbitals,
        spin_multiplicity=states["spin_multiplicity"],
        roots=states["roots"],
        weights=states["weights"],
    )
    supplied = raw.get("system", {})
    object_fields(supplied, SYSTEM_FIELDS, "system")
    for key, value in supplied.items():
        if key == "active_indices":
            require(active is not None, "Explicit active_indices require a prepared seed")
            require(
                orbital_indices(value, orbitals) == active,
                "system.active_indices disagree with the signed seed",
            )
        else:
            require(
                type(value) is type(system[key]) and value == system[key],
                "system." + key + " disagrees with initialization",
            )
    system["active_indices"] = [i + 1 for i in active] if active is not None else None
    system["orbital_index_base"] = 1
    result = dict(
        schema="pas.resolved-input.v1",
        name=raw["name"],
        profile=raw["profile"],
        input_file={"path": str(path), "sha256": input_digest},
        files=files,
        config=config,
        config_id=identity(config),
        policies=policies,
        system=system,
        initialization_mode=initialization_mode,
        paper_seed_id=case["seed_id"] if initialization_mode == "frozen_paper_seed" else None,
        historical_profile_unchanged=config == case["config"],
        paper_profile_unchanged=config == paper_config,
        paper_composition_purity_weight=case["paper_composition_purity_weight"],
    )
    result["resolved_id"] = identity(result)
    return result


def verify_resolved(record):
    require(
        record.get("resolved_id")
        == identity({k: v for k, v in record.items() if k != "resolved_id"}),
        "Resolved input was modified; create a new run for changed inputs",
    )
    for binding in record["files"].values():
        require(
            sha(binding["path"]) == binding["sha256"], "Run input file changed: " + binding["path"]
        )
    return record
