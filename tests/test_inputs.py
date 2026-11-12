"""Public input contracts: explicit defaults, one-based indices and strict JSON."""

import json
from copy import deepcopy

import pytest

from pas.common import identity
from pas_release.common import ValidationError, read_json
from pas_release.inputs import load_input, orbital_indices, verify_resolved
from pas_release.profiles import PAPER_INPUTS, check_paper_materials
from pas_release.resources import DATA, PROFILE_NAMES


def test_profile_uses_paper_alpha_without_other_configuration_changes(input_file):
    value = read_json(input_file)
    cases = read_json(DATA / "cases.json")["cases"]
    # paper changes only the two historical DMRG alphas. Aliases use the same defaults.
    for profile, key in PROFILE_NAMES.items():
        value["profile"] = profile
        expected = deepcopy(cases[key])
        expected["config"]["composition_purity_weight"] = 0.45
        for evidence in (
            None,
            {},
            dict(enabled=expected["config"]["channels"]),
            dict(weights=expected["config"]["weights"]),
            dict(
                enabled=list(reversed(expected["config"]["channels"])),
                weights={c: int(w) for c, w in expected["config"]["weights"].items()},
                composition_alpha=0.45,
            ),
        ):
            value.pop("evidence", None)
            if evidence is not None:
                value["evidence"] = evidence
            input_file.write_text(json.dumps(value))
            spec = load_input(input_file)
            assert spec["config"] == expected["config"], profile
            assert spec["config_id"] == identity(expected["config"]), profile
            assert spec["policies"] == expected["policies"], profile
            assert spec["historical_profile_unchanged"] is (key not in ("NX1_DMRG", "CI_OO_DMRG"))
            assert spec["paper_profile_unchanged"] is True
            assert spec["files"]["case_file"]["path"] == str(input_file.parent / "molecule.json")
            assert spec["system"]["orbital_index_base"] == 1


def test_evidence_reweighting_changes_only_weights_not_states_or_workflow(input_file):
    value = read_json(input_file)
    cases = read_json(DATA / "cases.json")["cases"]
    for profile in ("cas", "dmrg"):
        baseline = cases[PROFILE_NAMES[profile]]
        weights = {c: 1 for c in baseline["config"]["channels"]}
        weights["occupation"] = 2
        value.update(profile=profile, evidence=dict(weights=weights))
        input_file.write_text(json.dumps(value))
        spec = load_input(input_file)
        expected = deepcopy(baseline["config"])
        expected["composition_purity_weight"] = 0.45
        expected["weights"] = {c: float(w) for c, w in weights.items()}
        assert spec["config"] == expected
        assert spec["config_id"] == identity(expected) != identity(baseline["config"])
        assert spec["policies"] == baseline["policies"]
        assert spec["historical_profile_unchanged"] is False
        assert spec["system"]["weights"] == [0.5, 0.5]


def test_explicit_alpha_is_a_new_experiment_without_workflow_changes(input_file):
    value = read_json(input_file)
    historical = read_json(DATA / "cases.json")["cases"]["NX1_DMRG"]
    value.update(profile="dmrg", evidence=dict(composition_alpha=0.5))
    input_file.write_text(json.dumps(value))
    spec = load_input(input_file)
    assert spec["config"] == historical["config"]
    assert spec["policies"] == historical["policies"]
    assert spec["historical_profile_unchanged"] is True
    assert spec["paper_profile_unchanged"] is False


def test_paper_materials_cannot_be_omitted_or_replaced():
    files = deepcopy(PAPER_INPUTS["NX1_DMRG"])
    check_paper_materials("NX1_DMRG", files)
    for key in files:
        missing = {k: v for k, v in files.items() if k != key}
        with pytest.raises(ValidationError, match=key + ": required"):
            check_paper_materials("NX1_DMRG", missing)
        altered = {**files, key: dict(sha256="wrong")}
        with pytest.raises(ValidationError, match=key + ": differs"):
            check_paper_materials("NX1_DMRG", altered)


def test_invalid_evidence_is_rejected_before_reading_initialization(input_file):
    value = read_json(input_file)
    (input_file.parent / "molecule.json").unlink()
    invalid = [
        (dict(enabled=[]), "evidence.enabled"),
        (dict(enabled=["occupation", "occupation"]), "evidence.enabled"),
        (dict(enabled=["density"]), "evidence.enabled"),
        (dict(weights=dict(occupation=1)), "evidence.weights"),
        (dict(weights=dict(occupation=0, composition=1)), "evidence.weights.occupation"),
        (dict(weights=dict(occupation=True, composition=1)), "evidence.weights.occupation"),
        (dict(weights=dict(occupation=10**400, composition=1)), "evidence.weights.occupation"),
        (dict(weights=dict(occupation=1e308, composition=1e308)), "weight sum overflows"),
        *(
            (dict(composition_alpha=value), "evidence.composition_alpha")
            for value in (True, None, "0.45", -0.01, 1.01)
        ),
    ]
    for evidence, message in invalid:
        value["evidence"] = evidence
        input_file.write_text(json.dumps(value))
        with pytest.raises(ValidationError, match=message):
            load_input(input_file)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"schema": "unknown"}, "schema must be pas.input.v1"),
        ({"unused": True}, "input.unused: unknown field"),
        ({"settings": {"threads": True}}, "settings.threads: expected an integer"),
        ({"settings": {"threads": 33}}, "settings.threads: expected 1–32, got 33"),
        ({"settings": {"threds": 8}}, "Use 'threads' instead"),
        ({"settings": {"seed": -1}}, "settings.seed: fixed at 20261004"),
    ],
)
def test_invalid_public_input_is_rejected_before_execution(input_file, change, message):
    value = read_json(input_file)
    value.update(change)
    input_file.write_text(json.dumps(value))
    with pytest.raises(ValidationError, match=message):
        load_input(input_file)


@pytest.mark.parametrize(
    "states",
    [
        dict(roots=[1, 2], weights=[0.5, 0.4], spin_multiplicity=1),
        dict(roots=[1, 2], weights=[0.5, 0.5], spin_multiplicity=2),
    ],
)
def test_electron_spin_and_state_weights_are_checked(input_file, states):
    path = input_file.parent / "molecule.json"
    molecule = read_json(path)
    molecule["states"] = states
    path.write_text(json.dumps(molecule))
    with pytest.raises(ValidationError):
        load_input(input_file)


def test_invalid_xyz_is_rejected_before_execution(input_file):
    path = input_file.parent / "molecule.json"
    molecule = read_json(path)
    for geometry in ("H 0 0 0", "2\nH2\nH 0 0 0\n", "1\nH\nH NaN 0 0\n"):
        molecule["geometry_xyz"] = geometry
        path.write_text(json.dumps(molecule))
        with pytest.raises(ValidationError):
            load_input(input_file)


@pytest.mark.parametrize("text", ['{"schema":1,"schema":2}', '{"value":NaN}'])
def test_json_cannot_hide_duplicate_or_nonfinite_fields(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValidationError):
        read_json(path)


def test_one_based_indices_have_a_single_explicit_conversion():
    assert orbital_indices([1, 3, 4], 3) == [0, 2, 3]
    for invalid in ([0, 2, 3], [1, 1, 3], [True, 2, 3]):
        with pytest.raises(ValidationError):
            orbital_indices(invalid, 3)


def test_resolved_input_detects_changed_material_and_configuration(input_file):
    spec = load_input(input_file)
    verify_resolved(spec)
    modified = deepcopy(spec)
    modified["config"]["threads"] = 1
    with pytest.raises(ValidationError, match="modified"):
        verify_resolved(modified)
    (input_file.parent / "molecule.json").write_text("{}")
    with pytest.raises(ValidationError, match="changed"):
        verify_resolved(spec)
