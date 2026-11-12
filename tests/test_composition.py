# Selected from c06614a test_composition_pair.py; retention rationale: docs/maintenance/retention.json
"""Independent determinant identities and deterministic AO-channel construction."""

from dataclasses import replace
from itertools import combinations
from unittest.mock import patch

import numpy as np
import pytest

from pas.composition import _canonical_svd, _channel_matrix, determinant_pair_matrix
from pas.orbital_io import OpenMolcasHDF5Orbitals


def orbitals(cross=(0.4, 0.2), rotation=None):
    cross = np.diag(cross)
    n = cross.shape[0]
    s = np.block([[np.eye(n), cross], [cross.T, np.eye(n)]])
    values, vectors = np.linalg.eigh(s)
    c = vectors / np.sqrt(values) @ vectors.T
    if rotation is not None:
        c = c @ rotation
    return OpenMolcasHDF5Orbitals(
        c,
        s,
        np.zeros(2 * n),
        np.zeros(2 * n),
        ("s",) * (2 * n),
        ("C1", "O2"),
        np.array([[1 if i < n else 2, 1, 1, i % n] for i in range(2 * n)]),
        (2 * n,),
        {},
    )


@pytest.mark.parametrize("complex_values", [False, True])
def test_quadratic_equals_direct_determinant_for_all_subsets(complex_values):
    rng = np.random.default_rng(3405)
    sample = rng.normal(size=(6, 6))
    if complex_values:
        sample = sample + 1j * rng.normal(size=(6, 6))
    unitary, _ = np.linalg.qr(sample)
    z = unitary[:2]
    matrix = _channel_matrix(z)
    np.testing.assert_array_equal(np.diag(matrix), 0)
    for size in range(7):
        for active in combinations(range(6), size):
            selected = z[:, active]
            expected = np.linalg.det(selected @ selected.conj().T).real
            actual = matrix[np.ix_(active, active)].sum()
            assert actual == pytest.approx(expected, abs=2e-15)
            assert -1e-14 <= actual <= 1 + 1e-14
    assert matrix.sum() == pytest.approx(1)


def test_ao_constructor_matches_independent_two_center_determinants():
    rng = np.random.default_rng(710)
    rotation, _ = np.linalg.qr(rng.normal(size=(4, 4)))
    mo = orbitals(rotation=rotation)
    matrix, audit = determinant_pair_matrix(mo, [{"elements": ["C", "O"]}])
    pairs = []
    for i, strength in enumerate((0.4, 0.2)):
        first, second = (np.eye(4)[:, i], np.eye(4)[:, i + 2])
        pair = np.column_stack(
            (
                (first + second) / np.sqrt(2 * (1 + strength)),
                (first - second) / np.sqrt(2 * (1 - strength)),
            )
        )
        pairs.append(pair.T @ mo.ao_overlap @ mo.mo_coefficients)
    for active in combinations(range(4), 2):
        expected = sum(
            (
                w * np.linalg.det(z[:, active] @ z[:, active].T)
                for w, z in zip((0.4**2, 0.2**2), pairs)
            )
        ) / (0.4**2 + 0.2**2)
        assert matrix[np.ix_(active, active)].sum() == pytest.approx(expected)
    assert audit["channel_count"] == 2
    assert audit["target_ao_indices"] == [0, 1, 2, 3]
    assert audit["quadratic_convention"] == "xQx"
    assert audit["weight_sum"] == pytest.approx(0.2)
    altered = replace(mo, occupations=np.full(4, 1.0), type_indices=("i", "i", "s", "s"))
    np.testing.assert_array_equal(determinant_pair_matrix(altered, [{}])[0], matrix)


def test_degenerate_svd_uses_replayable_ao_gauge():
    cross = 0.3 * np.eye(3)
    expected = _canonical_svd(cross)
    original = np.linalg.svd
    rotation, _ = np.linalg.qr(np.random.default_rng(332).normal(size=(3, 3)))

    def rotated_svd(value, **kwargs):
        left, singular, right_t = original(value, **kwargs)
        return (left @ rotation, singular, rotation.T @ right_t)

    with patch("pas.composition.np.linalg.svd", side_effect=rotated_svd):
        actual = _canonical_svd(cross)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-14)
    np.testing.assert_allclose(actual[2], expected[2], atol=1e-14)
    assert actual[3] == [[0, 3]]
    mo = orbitals((0.3, 0.3, 0.3))
    baseline, audit = determinant_pair_matrix(mo, [{}])
    with patch("pas.composition.np.linalg.svd", side_effect=rotated_svd):
        replayed, replay = determinant_pair_matrix(mo, [{}])
    np.testing.assert_allclose(replayed, baseline, atol=1e-14)
    assert audit["channel_count"] == replay["channel_count"] == 3
    assert replay["center_pairs"][0]["canonical_blocks"] == [[0, 3]]
