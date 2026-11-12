"""Optional chemical preferences, bound to AO identities and the current MO basis."""

from __future__ import annotations

import math

import numpy as np

from .common import identity


def _rules(value):
    keys = {
        "center_indices",
        "centers",
        "elements",
        "radial_indices",
        "angular_momenta",
        "magnetic_quantum_numbers",
    }
    if not isinstance(value, list) or not value:
        raise ValueError("Preference rules must be a nonempty list of AO selectors")
    result = []
    for rule in value:
        if not isinstance(rule, dict) or not rule or set(rule) - keys:
            raise ValueError("Unknown or empty preference AO selector")
        item = {}
        for key, values in rule.items():
            if not isinstance(values, list) or not values:
                raise ValueError("Preference selector values must be nonempty lists")
            if key in ("centers", "elements"):
                if any(not isinstance(v, str) or not v.strip() for v in values):
                    raise ValueError("Preference atom labels/elements must be nonempty strings")
                values = [v.strip() for v in values]
            elif any(
                type(v) is not int
                or (key != "magnetic_quantum_numbers" and v < (1 if key == "radial_indices" else 0))
                for v in values
            ):
                raise ValueError("Invalid preference AO selector indices")
            item[key] = sorted(set(values))
        result.append(item)
    return result


def normalize_preferences(value):
    """Empty and zero-weight options disappear from effective run identities."""
    if value is None or value == {} or value == []:
        return None
    if not isinstance(value, dict) or set(value) - {"penalties", "rewards"}:
        raise ValueError("composition_preferences requires penalties/rewards lists")
    result = {}
    for kind in ("penalties", "rewards"):
        terms = value.get(kind)
        if terms is None:
            terms = []
        if not isinstance(terms, list):
            raise ValueError("Preference penalties/rewards must be lists")
        kept = []
        for term in terms:
            allowed = {"name", "weight", "rules"} | (
                {"center_pairs"} if kind == "rewards" else set()
            )
            if not isinstance(term, dict) or set(term) - allowed:
                raise ValueError("Unknown composition preference term fields")
            weight = term.get("weight")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
                or weight < 0
            ):
                raise ValueError("Preference weight must be finite and nonnegative")
            item = dict(weight=float(weight), rules=_rules(term.get("rules")))
            if "name" in term:
                if not isinstance(term["name"], str) or not term["name"].strip():
                    raise ValueError("Preference name must be a nonempty string")
                item["name"] = term["name"].strip()
            if "center_pairs" in term:
                pairs = term["center_pairs"]
                if (
                    not isinstance(pairs, list)
                    or not pairs
                    or any(
                        not isinstance(pair, list)
                        or len(pair) != 2
                        or any(type(i) is not int or i < 0 for i in pair)
                        or pair[0] == pair[1]
                        for pair in pairs
                    )
                ):
                    raise ValueError(
                        "Preference center_pairs require distinct zero-based atom pairs"
                    )
                canonical = sorted({tuple(sorted(pair)) for pair in pairs})
                if len(canonical) != len(pairs):
                    raise ValueError("Duplicate preference center_pairs")
                item["center_pairs"] = [list(pair) for pair in canonical]
            if weight:
                kept.append(item)
        if kept:
            result[kind] = kept
    total = sum(term["weight"] for terms in result.values() for term in terms)
    if not math.isfinite(1.0 + total):
        raise ValueError("Composition preference weights overflow")
    return result or None


def build_preferences(orbitals, specification):
    specification = normalize_preferences(specification)
    if specification is None:
        return None
    from .composition import determinant_pair_matrix
    from .orbital_io import ao_subspace_projection_scores, select_ao_indices

    projections, matrices, audits = [], [], dict(penalties=[], rewards=[])
    for term in specification.get("penalties", []):
        # Each selector must match; do not silently discard a mistyped subrule.
        selected = sorted(
            {i for rule in term["rules"] for i in select_ao_indices(orbitals, [rule])}
        )
        projections.append(
            ao_subspace_projection_scores(
                orbitals.mo_coefficients, orbitals.ao_overlap, selected
            ).tolist()
        )
        audits["penalties"].append(dict(ao_indices=selected))
    for term in specification.get("rewards", []):
        for rule in term["rules"]:
            select_ao_indices(orbitals, [rule])
        matrix, audit = determinant_pair_matrix(
            orbitals, term["rules"], center_pairs=term.get("center_pairs")
        )
        matrices.append(matrix.tolist())
        audits["rewards"].append(audit)
    return validate_preferences(
        dict(
            schema="pas_composition_preferences_v1",
            specification=specification,
            specification_id=identity(specification),
            penalty_projections=projections,
            reward_pairs=matrices,
            audit=audits,
        ),
        orbitals.orbital_count,
    )


def validate_preferences(data, nmo):
    if data is None:
        return None
    if (
        not isinstance(data, dict)
        or set(data)
        != {
            "schema",
            "specification",
            "specification_id",
            "penalty_projections",
            "reward_pairs",
            "audit",
        }
        or data["schema"] != "pas_composition_preferences_v1"
    ):
        raise ValueError("Invalid composition preference data")
    spec = normalize_preferences(data["specification"])
    if spec is None or spec != data["specification"] or identity(spec) != data["specification_id"]:
        raise ValueError("Composition preference specification binding differs")
    if len(data["penalty_projections"]) != len(spec.get("penalties", [])) or len(
        data["reward_pairs"]
    ) != len(spec.get("rewards", [])):
        raise ValueError("Missing composition preference measurements")
    from .metrics import validate_composition_pair

    for row in data["penalty_projections"]:
        row = np.asarray(row, float)
        if row.shape != (nmo,) or not np.isfinite(row).all() or np.any(row < 0) or np.any(row > 1):
            raise ValueError("Preference projections must cover the complete MO basis in [0,1]")
    for matrix in data["reward_pairs"]:
        matrix = validate_composition_pair(matrix, nmo)
        if not np.isclose(matrix.sum(), 1.0, atol=2e-7, rtol=0):
            raise ValueError("Preference pair coverage must use complete-basis normalization")
    return data


def assert_preference_binding(context, config):
    expected = normalize_preferences(config.get("composition_preferences"))
    data = getattr(context, "composition_preferences", None)
    if (data is None) != (expected is None) or (
        data is not None and data["specification"] != expected
    ):
        raise ValueError("Context composition preferences differ from the frozen configuration")


def preference_quadratic(data, nmo, size):
    """Encode the fixed offset as sum(x)/K; exact on the hard K-orbital domain."""
    validate_preferences(data, nmo)
    linear, pair, scale = np.zeros(nmo), np.zeros((nmo, nmo)), 1.0
    if data is not None:
        for term, projection in zip(
            data["specification"].get("penalties", []), data["penalty_projections"]
        ):
            linear += term["weight"] * ((1.0 - np.asarray(projection)) / size)
            scale += term["weight"]
        for term, matrix in zip(data["specification"].get("rewards", []), data["reward_pairs"]):
            pair += term["weight"] * np.asarray(matrix)
            scale += term["weight"]
    return linear, pair, scale


def preference_score(base, data, active, nmo):
    validate_preferences(data, nmo)
    value, scale = base, 1.0
    details = dict(base_composition=float(base), penalties=[], rewards=[])
    for term, projection in zip(
        data["specification"].get("penalties", []), data["penalty_projections"]
    ):
        measured = float(np.asarray(projection)[list(active)].mean())
        value += term["weight"] * (1.0 - measured)
        scale += term["weight"]
        details["penalties"].append(
            dict(
                name=term.get("name"),
                weight=term["weight"],
                projection=measured,
                deduction=term["weight"] * measured,
            )
        )
    for term, matrix in zip(data["specification"].get("rewards", []), data["reward_pairs"]):
        measured = float(np.asarray(matrix)[np.ix_(active, active)].sum())
        value += term["weight"] * measured
        scale += term["weight"]
        details["rewards"].append(
            dict(
                name=term.get("name"),
                weight=term["weight"],
                coverage=measured,
                reward=term["weight"] * measured,
            )
        )
    details.update(
        normalizer=scale,
        composition=float(value / scale),
        fixed_offset=sum(term["weight"] for term in data["specification"].get("penalties", [])),
        specification_id=data["specification_id"],
    )
    return float(value / scale), details
