"""Validated, solver-independent observations in one fixed real MO basis."""

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pas import composition, metrics
from pas.common import identity
from pas.config import CHANNELS, DEFAULTS
from pas.contracts import Context
from pas.orbital_io import ao_subspace_projection_scores, select_ao_indices

from .common import read_json, require, sha
from .inputs import object_fields, required_fields

SCHEMA = "pas.offline.v1"


def fields(value, allowed, required, location):
    object_fields(value, set(allowed), location)
    required_fields(value, set(required), location)


def integer(value, lower, upper, location):
    require(
        type(value) is int and lower <= value <= upper,
        f"{location}: expected integer {lower}..{upper}",
    )
    return value


def indices(value, count, location):
    require(
        isinstance(value, list) and value,
        location + ": expected a nonempty list of one-based indices",
    )
    result = [integer(i, 1, count, location) - 1 for i in value]
    require(len(set(result)) == len(result), location + ": duplicate index")
    return result


def array(value, shape, location):
    result = np.asarray(value)
    require(
        result.dtype.kind in "iuf" and result.shape == shape,
        f"{location}: expected a real numeric array of shape {shape}",
    )
    require(np.isfinite(result).all(), location + ": non-finite value")
    return result.astype(float)


def load_npz(path, expected_sha256):
    content = Path(path).read_bytes()
    require(
        hashlib.sha256(content).hexdigest() == expected_sha256,
        "Array file changed before reading: " + str(path),
    )
    with np.load(io.BytesIO(content), allow_pickle=False) as data:
        return {name: data[name].copy() for name in data.files}


@dataclass
class OfflineData:
    raw: dict
    files: list
    context: Context
    measured: list
    config: dict
    densities: np.ndarray | None
    entropy: np.ndarray | None
    mutual: np.ndarray | None
    audit: dict

    def unchanged(self):
        for item in self.files:
            require(
                sha(item["path"]) == item["sha256"],
                "Input changed during analysis: " + item["path"],
            )


def _evidence(raw):
    evidence = raw["evidence"]
    fields(evidence, ("enabled", "weights", "composition_alpha"), ("enabled",), "evidence")
    enabled = evidence["enabled"]
    require(
        isinstance(enabled, list)
        and enabled
        and all(isinstance(c, str) for c in enabled)
        and len(set(enabled)) == len(enabled)
        and set(enabled) <= set(CHANNELS),
        "evidence.enabled: choose distinct occupation, composition and/or qit",
    )
    enabled = [c for c in CHANNELS if c in enabled]
    weights = evidence.get("weights", {c: 1.0 for c in enabled})
    fields(weights, enabled, enabled, "evidence.weights")
    require(
        all(type(v) in (int, float) and np.isfinite(v) and v > 0 for v in weights.values())
        and np.isfinite(sum(weights.values())),
        "evidence.weights: positive finite weights required",
    )
    alpha = evidence.get("composition_alpha", 0.45)
    metrics.validate_composition_purity_weight(alpha, "determinant_pair_v1")
    require(
        "composition" in enabled or "composition_alpha" not in evidence,
        "evidence.composition_alpha requires the composition channel",
    )

    return enabled, weights, alpha


def _orbitals(raw, bind):
    orbital_spec = raw["orbitals"]
    fields(orbital_spec, ("format", "file"), ("format", "file"), "orbitals")
    orbital_file = bind(orbital_spec["file"])
    fmt = orbital_spec["format"]
    require(fmt in ("npz", "mokit_fch"), "orbitals.format: choose npz or mokit_fch")
    importer = {}
    if fmt == "npz":
        orbitals = load_npz(orbital_file["path"], orbital_file["sha256"])
    else:
        from .mokit_import import read_orbitals

        require(isinstance(raw["observations"], dict), "observations: expected a JSON object")
        orbitals, importer = read_orbitals(
            orbital_file["path"],
            density_section=raw["observations"].get("fch_density"),
            expected_sha256=orbital_file["sha256"],
        )
    required_fields(
        orbitals,
        (
            "coefficients",
            "overlap",
            "ao_centers",
            "ao_angular_momenta",
            "center_labels",
            "coordinates_angstrom",
        ),
        "orbital arrays",
    )
    coeff = np.asarray(orbitals["coefficients"])
    require(
        coeff.ndim == 2 and coeff.shape[0] == coeff.shape[1] and len(coeff),
        "orbitals.coefficients: this release requires a complete square real MO basis",
    )
    nmo = len(coeff)
    coeff = array(coeff, (nmo, nmo), "orbitals.coefficients")
    overlap = array(orbitals["overlap"], (nmo, nmo), "orbitals.overlap")
    require(
        np.allclose(overlap, overlap.T, atol=1e-10, rtol=0)
        and np.linalg.eigvalsh(overlap).min() > 0,
        "orbitals.overlap: expected a positive definite symmetric AO metric",
    )
    orthogonality = float(np.max(abs(coeff.T @ overlap @ coeff - np.eye(nmo))))
    require(orthogonality <= 1e-7, "Orbitals are not orthonormal in the supplied AO metric")
    labels = np.asarray(orbitals["center_labels"])
    require(
        labels.ndim == 1
        and len(labels)
        and labels.dtype.kind in "US"
        and all(str(v).strip() for v in labels),
        "orbitals.center_labels: expected atom labels",
    )
    coords = array(
        orbitals["coordinates_angstrom"], (len(labels), 3), "orbitals.coordinates_angstrom"
    )
    centers = np.asarray(orbitals["ao_centers"])
    angular = np.asarray(orbitals["ao_angular_momenta"])
    require(
        centers.shape == angular.shape == (nmo,)
        and centers.dtype.kind in "iu"
        and angular.dtype.kind in "iu"
        and np.all((centers >= 1) & (centers <= len(labels)))
        and np.all(angular >= 0),
        "AO metadata: one valid one-based center and angular momentum per AO required",
    )

    return SimpleNamespace(
        values=orbitals,
        binding=orbital_file,
        format=fmt,
        importer=importer,
        coefficients=coeff,
        overlap=overlap,
        labels=labels,
        coordinates=coords,
        centers=centers,
        angular=angular,
        orthogonality=orthogonality,
        nmo=nmo,
    )


def _observations(raw, bind, frame, enabled):
    nmo, fmt, orbitals, orbital_file = frame.nmo, frame.format, frame.values, frame.binding
    obs = raw["observations"]
    fields(
        obs,
        (
            "file",
            "orbitals",
            "electrons",
            "orbital_file_sha256",
            "density_convention",
            "entropy_unit",
            "mutual_information_convention",
            "fch_density",
            "natural_occupations",
        ),
        ("orbitals", "electrons", "orbital_file_sha256"),
        "observations",
    )
    require(
        obs["orbital_file_sha256"] == orbital_file["sha256"],
        "observations.orbital_file_sha256: observations are bound to a different orbital file",
    )
    arrays = {}
    if "file" in obs:
        observed_file = bind(obs["file"])
        arrays = load_npz(observed_file["path"], observed_file["sha256"])
    measured = indices(obs["orbitals"], nmo, "observations.orbitals")
    count = len(measured)
    nelec = integer(obs["electrons"], 0, 2 * count, "observations.electrons")
    states = raw["states"]
    fields(
        states,
        ("ids", "weights", "spin_multiplicity"),
        ("ids", "weights", "spin_multiplicity"),
        "states",
    )
    ids = states["ids"]
    require(
        isinstance(ids, list)
        and ids
        and all(isinstance(x, str) and x for x in ids)
        and len(set(ids)) == len(ids),
        "states.ids: distinct nonempty state labels required",
    )
    sw = array(states["weights"], (len(ids),), "states.weights")
    require(
        np.all(sw > 0) and np.isclose(sw.sum(), 1, atol=1e-12, rtol=0),
        "states.weights: positive weights must sum to 1",
    )
    spin = integer(states["spin_multiplicity"], 1, 2 * count + 1, "states.spin_multiplicity")
    require(
        (nelec - (spin - 1)) % 2 == 0 and spin - 1 <= min(nelec, 2 * count - nelec),
        "Observed electron count and spin are incompatible",
    )
    densities = entropy = mutual = None
    natural = "natural_occupations" in obs
    require(
        not natural or "occupation" in enabled,
        "observations.natural_occupations requires the occupation channel",
    )
    require(
        obs.get("density_convention") != "spin_summed_natural_orbital" or natural,
        "spin_summed_natural_orbital requires observations.natural_occupations",
    )
    if "occupation" in enabled:
        if natural:
            require(
                obs.get("density_convention") == "spin_summed_natural_orbital",
                "Natural occupations require density_convention=spin_summed_natural_orbital; "
                "the supplied orbital columns must be the corresponding natural orbitals",
            )
            require(
                "densities" not in arrays and "fch_density" not in obs,
                "Choose one density source: natural_occupations, densities or fch_density",
            )
            occupations = obs["natural_occupations"]
            require(
                isinstance(occupations, list)
                and all(
                    isinstance(row, list) and all(type(v) in (int, float) for v in row)
                    for row in occupations
                ),
                "observations.natural_occupations: expected numeric rows, one per state",
            )
            occupations = array(occupations, (len(ids), count), "observations.natural_occupations")
            densities = np.zeros((len(ids), count, count))
            densities[:, np.arange(count), np.arange(count)] = occupations
        else:
            require(
                obs.get("density_convention") == "spin_summed_mo",
                "observations.density_convention: expected spin_summed_mo",
            )
        if not natural and "fch_density" in obs:
            require(
                fmt == "mokit_fch" and len(ids) == 1 and "densities" not in arrays,
                "fch_density requires mokit_fch, one state, and no competing densities array",
            )
            from .mokit_import import read_density

            full = read_density(orbital_file["path"], obs["fch_density"], orbitals)
            arrays["densities"] = full[np.ix_(measured, measured)][None, :, :]
        if not natural:
            require(
                "densities" in arrays,
                "Enabled occupation evidence needs densities, fch_density or natural_occupations",
            )
            densities = array(
                arrays["densities"], (len(ids), count, count), "observations.densities"
            )
        metrics.validate_densities(densities, sw, nelec)
    else:
        require("fch_density" not in obs, "fch_density requires the occupation channel")
    if "qit" in enabled:
        require(
            obs.get("entropy_unit") == "natural_log"
            and obs.get("mutual_information_convention") == "half_mutual_information",
            "QIT requires entropy_unit=natural_log and mutual_information_convention=half_mutual_information",
        )
        required_fields(
            arrays, ("single_orbital_entropy", "mutual_information"), "Enabled qit evidence"
        )
        entropy = array(
            arrays["single_orbital_entropy"],
            (len(ids), count),
            "observations.single_orbital_entropy",
        )
        mutual = array(
            arrays["mutual_information"],
            (len(ids), count, count),
            "observations.mutual_information",
        )
        metrics.qit(entropy, mutual, sw)
        require(
            np.all(mutual <= np.minimum(entropy[:, :, None], entropy[:, None, :]) + 1e-8),
            "Half mutual information exceeds a single-orbital entropy",
        )

    return SimpleNamespace(
        indices=measured,
        electrons=nelec,
        densities=densities,
        entropy=entropy,
        mutual=mutual,
        states=dict(
            ids=ids, roots=list(range(1, len(ids) + 1)), weights=sw.tolist(), spin_multiplicity=spin
        ),
    )


def _context(raw, frame, observations, enabled):
    nmo, orbitals = frame.nmo, frame.values
    coeff, overlap, labels, coords = (
        frame.coefficients,
        frame.overlap,
        frame.labels,
        frame.coordinates,
    )
    centers, angular = frame.centers, frame.angular
    measured, nelec = observations.indices, observations.electrons
    count = len(measured)
    selection = raw["selection"]
    fields(
        selection,
        ("electrons", "orbitals", "reference_occupations", "initial_active", "required_active"),
        ("electrons", "orbitals", "reference_occupations", "initial_active"),
        "selection",
    )
    k = integer(selection["orbitals"], 2, count, "selection.orbitals")
    electrons = integer(selection["electrons"], 0, 2 * k, "selection.electrons")
    roles = selection["reference_occupations"]
    require(
        isinstance(roles, list)
        and len(roles) == nmo
        and all(type(r) is int and r in (0, 1, 2) for r in roles),
        "selection.reference_occupations: supply a reference 0/1/2 role for every MO; do not round correlated occupations",
    )
    require(
        sum(roles[i] for i in measured) == nelec,
        "Reference electron roles disagree with the measured electron scope",
    )
    active = indices(selection["initial_active"], nmo, "selection.initial_active")
    required = selection.get("required_active", [])
    require(isinstance(required, list), "selection.required_active: expected a list of indices")
    required = indices(required, nmo, "selection.required_active") if required else []
    require(
        len(active) == k and set(active) <= set(measured),
        "initial_active must contain K orbitals inside the measured scope",
    )
    outside = set(range(nmo)) - set(measured)
    require(
        not any(roles[i] == 1 for i in outside),
        "Measured scope must include every reference singly occupied orbital",
    )
    frozen = sorted(i for i in outside if roles[i] == 2)
    deleted = sorted(outside - set(frozen))
    inactive = sorted(i for i in measured if i not in active and roles[i] == 2)
    if "electron_count" in orbitals:
        require(
            sum(roles) == orbitals["electron_count"],
            "Reference electron roles disagree with the orbital file electron count",
        )
    projection = pair = None
    composition_audit = None
    if "composition" in enabled:
        target = raw.get("target")
        fields(target, ("atoms", "angular_momenta"), ("atoms",), "target")
        atoms = indices(target["atoms"], len(labels), "target.atoms")
        rules = {"center_indices": atoms}
        if "angular_momenta" in target:
            values = target["angular_momenta"]
            require(
                isinstance(values, list)
                and values
                and all(type(x) is int and x >= 0 for x in values),
                "target.angular_momenta: nonnegative integer list required",
            )
            rules["angular_momenta"] = values
        # Only center and angular momentum columns are consumed by these rules.
        frame = SimpleNamespace(
            mo_coefficients=coeff,
            ao_overlap=overlap,
            orbital_count=nmo,
            basis_function_ids=np.column_stack(
                (centers, np.zeros(nmo, int), angular, np.zeros(nmo, int))
            ),
            center_labels=labels.tolist(),
        )
        selected = select_ao_indices(frame, [rules])
        projection = ao_subspace_projection_scores(coeff, overlap, selected).tolist()
        pair, composition_audit = composition.determinant_pair_matrix(frame, [rules])
        pair = pair.tolist()
        composition_audit = dict(
            model="determinant_pair_v1",
            target_ao_count=len(selected),
            target_atom_indices=[i + 1 for i in atoms],
            pair_channel_count=len(composition_audit["channels"]),
        )
    else:
        require("target" not in raw, "target requires the composition channel")
    context = Context(
        basis_id=identity(dict(coefficients=coeff.tolist(), overlap=overlap.tolist())),
        geometry_id=identity(dict(labels=labels.tolist(), coordinates_angstrom=coords.tolist())),
        backend="external_observations",
        nmo=nmo,
        active=tuple(active),
        inactive=tuple(inactive),
        total_electrons=sum(roles),
        electrons=electrons,
        frozen=tuple(frozen),
        deleted=tuple(deleted),
        electron_roles=tuple(roles),
        required_active=tuple(required),
        states=observations.states,
        composition=projection,
        composition_pair=pair,
    )
    context.partition(active)
    return context, composition_audit


def _settings(raw, enabled, weights, alpha):
    search = raw.get("search", {})
    fields(
        search,
        ("seed", "exact_limit", "annealing_steps", "annealing_restarts", "candidates"),
        (),
        "search",
    )
    limits = dict(
        seed=(0, 2**32 - 1),
        exact_limit=(1, 1000000),
        annealing_steps=(1, 10000000),
        annealing_restarts=(1, 100),
        candidates=(1, 100),
    )
    config = {
        key: integer(
            search.get(key, 8 if key == "candidates" else DEFAULTS[key]), *limit, "search." + key
        )
        for key, limit in limits.items()
    }
    config.update(
        channels=enabled,
        weights=weights,
        composition_purity_weight=float(alpha),
        composition_model="determinant_pair_v1",
        proposal_score_model="terminal_aligned_v1",
    )
    return config


def load(path):
    path = Path(path).resolve()
    files = []

    def bind(value):
        require(isinstance(value, str) and value.strip(), "Expected a nonempty input file path")
        item = (path.parent / value).resolve()
        require(item.is_file(), "Input file is missing: " + str(item))
        binding = dict(path=str(item), sha256=sha(item))
        files.append(binding)
        return binding

    input_file = bind(str(path))
    raw = read_json(path, expected_sha256=input_file["sha256"])
    fields(
        raw,
        (
            "schema",
            "name",
            "source",
            "orbitals",
            "observations",
            "selection",
            "states",
            "evidence",
            "target",
            "search",
        ),
        ("schema", "name", "source", "orbitals", "observations", "selection", "states", "evidence"),
        "input",
    )
    require(raw["schema"] == SCHEMA, "input.schema: expected " + SCHEMA)
    require(isinstance(raw["name"], str) and raw["name"].strip(), "input.name: required")
    fields(
        raw["source"],
        ("program", "version", "method", "description"),
        ("program", "method"),
        "source",
    )
    require(
        all(isinstance(v, str) and v.strip() for v in raw["source"].values()),
        "source: use nonempty strings",
    )

    enabled, weights, alpha = _evidence(raw)
    frame = _orbitals(raw, bind)
    observations = _observations(raw, bind, frame, enabled)
    context, composition_audit = _context(raw, frame, observations, enabled)
    config = _settings(raw, enabled, weights, alpha)
    audit = dict(
        orthogonality_max_error=frame.orthogonality,
        composition=composition_audit,
        importer=frame.importer,
        scope="fixed_orbital_observations_not_reoptimized_target_CAS",
        density_convention="spin_summed_mo" if observations.densities is not None else None,
        measured_electrons=observations.electrons,
        orbital_indices_are_one_based=True,
    )
    if "natural_occupations" in raw["observations"]:
        audit["density_input"] = dict(
            format="natural_occupations",
            basis="input_declares_corresponding_natural_orbitals_for_each_state",
            construction="diagonal_in_supplied_orbital_order",
            natural_occupations=raw["observations"]["natural_occupations"],
        )
    spec = OfflineData(
        raw,
        files,
        context,
        observations.indices,
        config,
        observations.densities,
        observations.entropy,
        observations.mutual,
        audit,
    )
    spec.unchanged()
    return spec
