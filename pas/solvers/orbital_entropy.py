"""Audited real-SZ orbital entropies from the native Block2 NPDM operators.

Operator assembly and eigensolvers are reused from Block2 0.5.3's
OrbitalEntropy API. Only the probability-to-entropy numerical boundary differs.
"""

from __future__ import annotations

import numpy as np

PSD_TOLERANCE = 1e-12
TRACE_TOLERANCE = 1e-10
CONSISTENCY_TOLERANCE = 1e-10


def entropy_from_spectra(spectra):
    """Evaluate -tr(rho log rho), rejecting more than roundoff repair."""
    raw = np.asarray(spectra)
    if np.iscomplexobj(raw):
        raise RuntimeError("Orbital RDM spectrum is unexpectedly complex")
    values = np.asarray(raw, dtype=float)
    if not values.size or values.ndim < 1 or not np.all(np.isfinite(values)):
        raise RuntimeError("Orbital RDM spectrum contains non-finite or missing values")
    minimum = float(values.min())
    negative_mass = -np.minimum(values, 0).sum(axis=-1)
    if minimum < -PSD_TOLERANCE or np.max(negative_mass) > PSD_TOLERANCE:
        raise RuntimeError(f"Orbital RDM is not positive semidefinite: min={minimum:.17g}")
    traces = values.sum(axis=-1)
    trace_error = float(np.max(abs(traces - 1)))
    if trace_error > TRACE_TOLERANCE:
        raise RuntimeError(f"Orbital RDM does not have unit trace: error={trace_error:.17g}")
    probabilities = np.maximum(values, 0)
    positive_traces = probabilities.sum(axis=-1, keepdims=True)
    probabilities = probabilities / positive_traces
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = -probabilities[positive] * np.log(probabilities[positive])
    return terms.sum(axis=-1), dict(
        schema="pas_orbital_entropy_numerics_v1",
        psd_absolute_tolerance=PSD_TOLERANCE,
        trace_absolute_tolerance=TRACE_TOLERANCE,
        minimum_raw_eigenvalue=minimum,
        maximum_raw_trace_error=trace_error,
        negative_eigenvalue_count=int(np.count_nonzero(values < 0)),
        maximum_clipped_negative_mass=float(np.max(negative_mass)),
        maximum_normalization_correction=float(np.max(abs(positive_traces - 1))),
        correction="clip_roundoff_negatives_then_unit_trace",
    )


def orbital_entropies(driver, ket, *, orb_type):
    """Use native masked contractions, without library-wide monkeypatches."""
    from pyblock2.driver.core import OrbitalEntropy, SymmetryTypes

    if (
        driver.bw.symm_type != SymmetryTypes.SZ
        or driver.reorder_idx is not None
        or orb_type not in (1, 2)
    ):
        raise RuntimeError("Orbital entropy requires real SZ in unreordered DMRG site order")
    count = driver.n_sites
    shape = (count,) * orb_type
    exprs, width = (
        OrbitalEntropy.get_one_orb_rdm_exprs()
        if orb_type == 1
        else OrbitalEntropy.get_two_orb_rdm_exprs()
    )
    copied = ket.deep_copy(ket.info.tag + "@PAS-ORB-ENT-TMP")
    pdms = driver.get_npdm(
        copied,
        pdm_type=[len(expr) // 2 for expr, _ in exprs],
        npdm_expr=[expr for expr, _ in exprs],
        mask=[list(mask) for _, mask in exprs],
        iprint=0,
    )
    packed = np.zeros(shape + (width,), dtype=float)
    for ((_, mask), terms), pdm in zip(exprs.items(), pdms, strict=True):
        pdm = np.asarray(pdm)
        if np.iscomplexobj(pdm) or not np.all(np.isfinite(pdm)):
            raise RuntimeError("Invalid real orbital entropy NPDM")
        if orb_type == 2:
            if not mask:
                pdm = pdm[None, None]
            elif len(set(mask)) == 1:
                pdm = pdm[:, None] if mask[0] == 0 else pdm[None, :]
        for slot, factor in terms:
            packed[..., slot] += pdm * factor
    if orb_type == 1:
        spectra = packed
    else:
        spectra = np.empty(shape + (16,))
        for i, j in np.ndindex(shape):
            spectra[i, j] = OrbitalEntropy.get_two_orb_rdm_eigvals(packed[i, j], diag_only=i == j)
    return entropy_from_spectra(spectra)
