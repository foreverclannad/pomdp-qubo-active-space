# Selected from c06614a test_orbital_entropy_numerics.py; retention rationale: docs/maintenance/retention.json
"""Entropy boundary regressions, including the native NPDM call site."""

import numpy as np
import pytest

TIF3_PAIR_SPECTRUM = np.array(
    [
        -1.554312234475219e-14,
        8.374476217847691e-09,
        6.257457330646822e-08,
        1.0000003799633918e-08,
        7.326775411137579e-08,
        4.093629493717721e-06,
        4.017209564230427e-06,
        4.954113041950825e-06,
        1.608127044820706e-05,
        0.000564555378254304,
        3.957332349591169e-06,
        0.0001907615038442868,
        0.0030397034379757125,
        0.0002002001630710604,
        0.002510115279153715,
        0.9934614064660113,
    ]
)


@pytest.mark.parametrize("size", [4, 16])
def test_exact_zero_pure_and_maximally_mixed_spectra(size):
    from pas.solvers.orbital_entropy import entropy_from_spectra

    pure = np.zeros(size)
    pure[-1] = 1
    values, _ = entropy_from_spectra(np.stack([pure, np.full(size, 1 / size)]))
    np.testing.assert_allclose(values, [0, np.log(size)], atol=1e-15, rtol=0)


@pytest.mark.parametrize(
    "values, message",
    [
        ([-1e-07, 0, 0.5, 0.5000001], "positive semidefinite"),
        ([0, 0, 0.4, 0.5], "unit trace"),
        ([0, 0, np.nan, 1], "non-finite"),
        ([0, 0, np.inf, 1], "non-finite"),
        ([0, 0, 0.5 + 0.01j, 0.5 - 0.01j], "complex"),
    ],
)
def test_invalid_rdm_spectra_fail_closed(values, message):
    from pas.solvers.orbital_entropy import entropy_from_spectra

    with pytest.raises(RuntimeError, match=message):
        entropy_from_spectra(np.asarray(values)[None, :])
