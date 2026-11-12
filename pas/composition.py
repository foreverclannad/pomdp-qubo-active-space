"""Determinant coverage of general target-AO pairs, exactly quadratic in MO selection.

The channel construction uses only AO identities and their metric.  It does not
use occupations, reference spaces, bond lists, or a fitted ranking model.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from .orbital_io import select_ao_indices

MINIMUM_CHANNEL_OVERLAP = 1e-6
METRIC_RCOND = 1e-10
DEGENERACY_TOLERANCE = 1e-12
ORTHONORMALITY_TOLERANCE = 1e-7


def _inverse_sqrt(metric):
    values, vectors = np.linalg.eigh(metric)
    cutoff = METRIC_RCOND * max(float(values.max()), 1.0)
    if values.min() < -cutoff:
        raise ValueError("Target AO centre metric is not positive semidefinite")
    keep = values > cutoff
    if not keep.any():
        raise ValueError("Target AO centre metric has no supported directions")
    return (vectors[:, keep] / np.sqrt(values[keep])) @ vectors[:, keep].T


def _coordinate_basis(span):
    """Fix a subspace gauge by projected AO axes in their native index order."""
    projector = span @ span.T
    basis = []
    for column in projector.T:
        vector = column.copy()
        for _ in range(2):
            for previous in basis:
                vector -= previous * (previous @ vector)
        norm = np.linalg.norm(vector)
        if norm > METRIC_RCOND:
            basis.append(vector / norm)
        if len(basis) == span.shape[1]:
            return np.column_stack(basis)
    raise ValueError("Cannot resolve the target-channel subspace gauge")


def _canonical_svd(cross):
    left, values, right_t = np.linalg.svd(cross, full_matrices=False)
    right = right_t.T.copy()
    groups = []
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and abs(values[end] - values[start]) <= DEGENERACY_TOLERANCE:
            end += 1
        old = left[:, start:end].copy()
        fixed = _coordinate_basis(old)
        rotation = old.T @ fixed
        left[:, start:end] = fixed
        right[:, start:end] = right[:, start:end] @ rotation
        groups.append([start, end])
        start = end
    return left, values, right, groups


def _channel_matrix(amplitudes):
    """Return Q with x@Q@x=det(Z[:,A]@Z[:,A].H), including complex Z."""
    z = np.asarray(amplitudes)
    if z.ndim != 2 or z.shape[0] != 2 or not np.isfinite(z).all():
        raise ValueError("Pair amplitudes must be a finite two-row matrix")
    if not np.allclose(z @ z.conj().T, np.eye(2), atol=ORTHONORMALITY_TOLERANCE, rtol=0):
        raise ValueError("Target pair is not orthonormal in the complete MO basis")
    wedge = np.outer(z[0], z[1]) - np.outer(z[1], z[0])
    matrix = np.abs(wedge) ** 2 / 2
    np.fill_diagonal(matrix, 0)
    return matrix


def _target_pair_data(orbitals, target_rules, *, center_pairs=None):
    """Return (Q, audit), where Q is symmetric, zero-diagonal and x@Q@x=Bdet.

    Every supported target-AO centre pair contributes principal-overlap channels.
    Within a numerically degenerate singular block, projected native AO axes fix
    the left gauge; the same rotation is applied on the right.  Actual overlaps
    determine normalization and squared channel weights.  This is a recorded
    convention, not a claim of invariance under regrouping different channels.
    """
    c = np.asarray(orbitals.mo_coefficients)
    s = np.asarray(orbitals.ao_overlap)
    nmo = orbitals.orbital_count
    if not np.isfinite(c).all() or not np.isfinite(s).all():
        raise ValueError("Orbital coefficients and AO metric must be finite")
    if not np.allclose(c.T @ s @ c, np.eye(nmo), atol=ORTHONORMALITY_TOLERANCE, rtol=0):
        raise ValueError("MO basis is not AO-metric orthonormal")
    selected = select_ao_indices(orbitals, target_rules)
    centers = {}
    for index in selected:
        center = int(orbitals.basis_function_ids[index, 0]) - 1
        centers.setdefault(center, []).append(index)
    requested = None if center_pairs is None else {tuple(sorted(pair)) for pair in center_pairs}
    if requested is not None and (
        not requested
        or any(
            len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= centers.keys()
            for pair in requested
        )
    ):
        raise ValueError("Requested preference atom pair is absent from the selected AO centers")
    inverse = {
        center: _inverse_sqrt(s[np.ix_(indices, indices)]) for center, indices in centers.items()
    }
    matrix = np.zeros((nmo, nmo))
    channels, center_pairs, plus, minus = [], [], [], []
    weight_sum = 0.0
    for first, second in combinations(sorted(centers), 2):
        if requested is not None and (first, second) not in requested:
            continue
        ai, aj = centers[first], centers[second]
        cross = inverse[first] @ s[np.ix_(ai, aj)] @ inverse[second]
        left, singular, right, groups = _canonical_svd(cross)
        center_pairs.append(
            dict(
                centers=[first, second], singular_values=singular.tolist(), canonical_blocks=groups
            )
        )
        for index, strength in enumerate(singular):
            if strength < MINIMUM_CHANNEL_OVERLAP or strength >= 1 - METRIC_RCOND:
                continue
            fi, fj = np.zeros(nmo), np.zeros(nmo)
            fi[ai] = inverse[first] @ left[:, index]
            fj[aj] = inverse[second] @ right[:, index]
            overlap = float(fi @ s @ fj)
            if overlap < 0:
                fj *= -1
                overlap = -overlap
            if not MINIMUM_CHANNEL_OVERLAP <= overlap < 1 - METRIC_RCOND:
                raise ValueError("Canonical target channel changed its overlap eligibility")
            pair = np.column_stack(
                ((fi + fj) / np.sqrt(2 * (1 + overlap)), (fi - fj) / np.sqrt(2 * (1 - overlap)))
            )
            pair_error = float(np.max(np.abs(pair.T @ s @ pair - np.eye(2))))
            if pair_error > ORTHONORMALITY_TOLERANCE:
                raise ValueError("Target AO pair is not metric orthonormal")
            amplitudes = pair.T @ s @ c
            plus.append((np.abs(amplitudes[0]) ** 2).tolist())
            minus.append((np.abs(amplitudes[1]) ** 2).tolist())
            weight = overlap**2
            matrix += weight * _channel_matrix(amplitudes)
            weight_sum += weight
            channels.append(
                dict(
                    centers=[first, second],
                    principal_channel=index,
                    principal_overlap=overlap,
                    weight=weight,
                    pair_orthonormality_max_error=pair_error,
                )
            )
    if not channels or not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("No supported target-AO pairing channels")
    if requested is not None and requested - {tuple(channel["centers"]) for channel in channels}:
        raise ValueError("A requested preference atom pair has no supported overlap channel")
    matrix /= weight_sum
    if not np.isfinite(matrix).all() or not np.array_equal(matrix, matrix.T):
        raise ValueError("Invalid determinant-pair quadratic matrix")
    if not np.isclose(matrix.sum(), 1.0, atol=2 * ORTHONORMALITY_TOLERANCE, rtol=0):
        raise ValueError("Complete MO space does not cover the target pairs")
    audit = dict(
        schema="determinant_pair_v1",
        quadratic_convention="xQx",
        target_ao_indices=list(selected),
        channel_count=len(channels),
        minimum_channel_overlap=MINIMUM_CHANNEL_OVERLAP,
        metric_rcond=METRIC_RCOND,
        degeneracy_tolerance=DEGENERACY_TOLERANCE,
        orthonormality_tolerance=ORTHONORMALITY_TOLERANCE,
        degeneracy_gauge="left_projected_native_AO_axes_double_Gram_Schmidt_v1",
        weight_rule="squared_actual_channel_overlap_normalized_over_all_supported_channels",
        weight_sum=weight_sum,
        center_pairs=center_pairs,
        channels=channels,
    )
    harmonic_channels = dict(
        schema="target_ao_harmonic_channels_v1",
        plus=plus,
        minus=minus,
        weights=[channel["weight"] / weight_sum for channel in channels],
    )
    return matrix, harmonic_channels, audit


def determinant_pair_matrix(orbitals, target_rules, *, center_pairs=None):
    """Return the exact determinant Q and unchanged legacy construction audit."""
    matrix, _, audit = _target_pair_data(orbitals, target_rules, center_pairs=center_pairs)
    return matrix, audit


def harmonic_channel_data(orbitals, target_rules):
    """Serializable harmonic channels in the same fixed target-AO gauge.

    Each plus/minus row contains squared overlaps with the complete MO basis.
    No occupations, reference space or chosen chemical bond enters this data.
    """
    _, channels, audit = _target_pair_data(orbitals, target_rules)
    audit = dict(
        audit,
        schema="harmonic_channel_pair_v1",
        coverage_formula="sum_normalized_overlap_squared_H_plus_minus",
    )
    audit.pop("quadratic_convention")
    return channels, audit
