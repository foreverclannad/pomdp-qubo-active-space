"""External observations: physical scoring, input boundaries and optional FCH I/O."""

import copy
import itertools
import json

import numpy as np
import pytest

from pas_release import offline
from pas_release.cli import main
from pas_release.common import compare_values, sha
from pas_release.offline_input import load, load_npz


def prepared(tmp_path):
    directory = tmp_path / "example"
    offline.example(directory)
    path = directory / "input.json"
    return path, json.loads(path.read_text())


def save(path, raw):
    path.write_text(json.dumps(raw))


def test_reviewer_example_cli_and_reference(tmp_path, capsys):
    path, _ = prepared(tmp_path)
    assert main(["offline", "check", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["legal_candidates"] == 4
    output = tmp_path / "result"
    assert main(["offline", "select", str(path), "--output", str(output), "--json"]) == 0
    actual = json.loads(capsys.readouterr().out)
    expected = json.loads((path.parent / "expected.json").read_text())
    tolerance = expected.pop("tolerance")
    compare_values({key: actual[key] for key in expected}, expected, tolerance)
    assert actual["new_quantum_chemistry_calculations"] == 0
    assert actual["optimized_candidate_energy_hartree"] is None
    assert len((output / "candidates.csv").read_text().splitlines()) == 5
    assert all(i >= 1 for row in actual["solver"]["visited_candidates"] for i in row)
    with pytest.raises(ValueError, match="new result directory"):
        offline.select(path, output)


def test_channel_masks_match_independent_enumeration(tmp_path):
    path, original = prepared(tmp_path)
    observations = path.parent / "observations.npz"
    arrays = load_npz(observations, sha(observations))
    permutation = [2, 0, 3, 1]
    # Two distinct, valid fixed-basis wavefunctions: the second is obtained by
    # permuting fermionic modes of the first, consistently in all observations.
    for key in ("densities", "mutual_information"):
        arrays[key] = np.concatenate((arrays[key], arrays[key][:, permutation][:, :, permutation]))
    arrays["single_orbital_entropy"] = np.concatenate(
        (arrays["single_orbital_entropy"], arrays["single_orbital_entropy"][:, permutation])
    )
    np.savez_compressed(observations, **arrays)
    state_weights = np.array([0.25, 0.75])
    original["states"].update(ids=["original", "mode-permuted"], weights=state_weights.tolist())
    for count in (1, 2, 3):
        for channels in itertools.combinations(("occupation", "composition", "qit"), count):
            raw = copy.deepcopy(original)
            weights = {c: float(i + 1) for i, c in enumerate(channels)}
            raw["evidence"] = dict(enabled=list(channels), weights=weights)
            if "composition" not in channels:
                raw.pop("target")
            save(path, raw)
            spec = load(path)
            q = offline.objective(spec)
            for first in (0, 1):
                for second in (2, 3):
                    active = [first, second]
                    terms = {}
                    if "occupation" in channels:
                        spectrum = np.linalg.eigvalsh(spec.densities[:, active][:, :, active])
                        terms["occupation"] = (
                            state_weights @ np.sum(spectrum * (2 - spectrum), axis=1) / 2
                        )
                    if "composition" in channels:
                        terms["composition"] = (
                            0.45 * np.mean(np.array(spec.context.composition)[active])
                            + 0.55
                            * np.array(spec.context.composition_pair)[np.ix_(active, active)].sum()
                        )
                    if "qit" in channels:
                        outside = [i for i in range(4) if i not in active]
                        internal = spec.mutual[:, first, second]
                        boundary = spec.mutual[:, active][:, :, outside].sum(axis=(1, 2))
                        terms["qit"] = (
                            state_weights
                            @ (np.mean(spec.entropy[:, active], axis=1) + internal - boundary)
                            / (2 * np.log(4))
                        )
                    expected = sum(weights[c] * terms[c] for c in channels) / sum(weights.values())
                    assert q(active) == pytest.approx(expected, abs=1e-12)
                    assert offline.score(spec, active)["selection_score"] == pytest.approx(
                        expected, abs=1e-12
                    )


def test_malformed_observations_fail_before_output(tmp_path):
    path, raw = prepared(tmp_path)
    bad = [
        ("binding", lambda x: x["observations"].update(orbital_file_sha256="0" * 64)),
        ("QIT", lambda x: x["observations"].update(entropy_unit="log2")),
        ("weights", lambda x: x["evidence"]["weights"].update(qit=True)),
        ("electrons", lambda x: x["observations"].update(electrons=2)),
        ("required_active", lambda x: x["selection"].update(required_active=False)),
    ]
    for name, change in bad:
        altered = copy.deepcopy(raw)
        change(altered)
        save(path, altered)
        output = tmp_path / name
        with pytest.raises(ValueError):
            offline.select(path, output)
        assert not output.exists()
    save(path, raw)
    observations = path.parent / "observations.npz"
    arrays = load_npz(observations, sha(observations))
    arrays.pop("mutual_information")
    np.savez_compressed(observations, **arrays)
    with pytest.raises(ValueError, match="qit"):
        offline.select(path, tmp_path / "missing")
    assert not (tmp_path / "missing").exists()
    # Valid external text precision near the physical occupation bounds must
    # agree with the core's tolerance, including clipping of its final score.
    raw["evidence"] = dict(enabled=["occupation"])
    raw.pop("target")
    save(path, raw)
    arrays["densities"] = np.diag([2 + 9e-8, 2 + 9e-8, -9e-8, -9e-8])[None, :, :]
    np.savez_compressed(observations, **arrays)
    result = offline.select(path, tmp_path / "density-roundoff")
    assert result["recommended"]["selection_score"] == 0
    assert 1e-7 < result["audit"]["score_evaluation_max_error"] < 1e-6


def test_orbital_permutation_preserves_physical_choice(tmp_path):
    path, raw = prepared(tmp_path)
    before = offline.select(path, tmp_path / "before")
    permutation = np.array([2, 0, 3, 1])
    inverse = np.argsort(permutation)
    orbitals = path.parent / "orbitals.npz"
    arrays = load_npz(orbitals, sha(orbitals))
    arrays["coefficients"] = arrays["coefficients"][:, permutation]
    np.savez_compressed(orbitals, **arrays)
    observations = path.parent / "observations.npz"
    arrays = load_npz(observations, sha(observations))
    for key in ("densities", "mutual_information"):
        arrays[key] = arrays[key][:, permutation][:, :, permutation]
    arrays["single_orbital_entropy"] = arrays["single_orbital_entropy"][:, permutation]
    np.savez_compressed(observations, **arrays)
    raw["selection"]["reference_occupations"] = np.array(raw["selection"]["reference_occupations"])[
        permutation
    ].tolist()
    raw["selection"]["initial_active"] = (
        inverse[np.array(raw["selection"]["initial_active"]) - 1] + 1
    ).tolist()
    raw["observations"]["orbital_file_sha256"] = sha(orbitals)
    save(path, raw)
    after = offline.select(path, tmp_path / "after")
    restored = sorted(permutation[np.array(after["recommended"]["active_indices"]) - 1] + 1)
    assert restored == before["recommended"]["active_indices"]
    assert after["recommended"]["selection_score"] == pytest.approx(
        before["recommended"]["selection_score"], abs=1e-12
    )


def test_inputs_edited_during_selection_are_rejected(tmp_path, monkeypatch):
    path, _ = prepared(tmp_path)
    solve = offline.qubo.solve

    def changed(*args, **kwargs):
        value = solve(*args, **kwargs)
        path.write_text(path.read_text() + "\n")
        return value

    monkeypatch.setattr(offline.qubo, "solve", changed)
    with pytest.raises(ValueError, match="Input changed"):
        offline.select(path, tmp_path / "result")
    assert not (tmp_path / "result").exists()


@pytest.mark.mokit
def test_mokit_example_matches_arrays_without_modifying_input(tmp_path):
    pytest.importorskip("mokit")
    pytest.importorskip("pyscf")
    path, _ = prepared(tmp_path)
    before = {p.name: sha(p) for p in path.parent.iterdir() if p.is_file()}
    plain = offline.select(path, tmp_path / "plain")
    converted = offline.select(path.with_name("input-mokit.json"), tmp_path / "converted")
    assert converted["audit"]["importer"]["program"] == "MOKIT"
    for key in ("initial", "recommended", "candidates"):
        compare_values(converted[key], plain[key], tolerance=1e-7)
    assert before == {p.name: sha(p) for p in path.parent.iterdir() if p.is_file()}


@pytest.mark.mokit
def test_mokit_spherical_d_f_density_roundtrip(tmp_path):
    pytest.importorskip("mokit")
    pytest.importorskip("pyscf")
    from mokit.lib.py2fch import write_pyscf_dm_into_fch
    from mokit.lib.py2fch_direct import fchk
    from pyscf import gto, lib, scf

    from pas_release.mokit_import import read_orbitals

    lib.num_threads(1)
    mol = gto.M(atom="Ne 0 0 0", basis="cc-pvtz", verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    assert mf.converged
    n = mol.nao_nr()
    rotation, _ = np.linalg.qr(np.random.default_rng(41).normal(size=(n, n)))
    occupations = np.array([2.0] * 4 + [1.2, 0.8] + [0.0] * (n - 6))
    density = (rotation * occupations) @ rotation.T
    fch = tmp_path / "ne.fch"
    fchk(mf, str(fch), density=True)
    write_pyscf_dm_into_fch(str(fch), 1, n, mf.mo_coeff @ density @ mf.mo_coeff.T, False)
    before = sha(fch)
    arrays, _ = read_orbitals(fch, density_section="Total SCF Density", expected_sha256=before)
    assert 2 in arrays["ao_angular_momenta"] and 3 in arrays["ao_angular_momenta"]
    np.testing.assert_allclose(arrays["coefficients"], mf.mo_coeff, atol=1e-7, rtol=0)
    np.testing.assert_allclose(arrays["overlap"], mf.get_ovlp(), atol=1e-7, rtol=0)
    np.testing.assert_allclose(arrays["density_mo"], density, atol=1e-7, rtol=0)
    assert sha(fch) == before
