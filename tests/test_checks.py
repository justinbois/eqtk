import numpy as np
import pandas as pd
import pytest

import eqtk

import hypothesis
import hypothesis.strategies as hs
import hypothesis.extra.numpy as hnp


def simple_binding(cA0, cB0, Kd):
    """
    Compute equilibrium for simple binding, AB <=> A + B.
    c0 = (A, B)
    """
    b = Kd + cA0 + cB0
    if cA0 == 0 or cB0 == 0:
        cAB = 0
    else:
        # Use roots function to avoid precision errors in quadratic formula
        poly_coeffs = np.array([1.0, -(Kd + cA0 + cB0), cA0 * cB0])
        cAB = np.roots(poly_coeffs).min()

    cA = cA0 - cAB
    cB = cB0 - cAB

    # Return problem for checking
    c0 = np.array([cA0, cB0, 0], dtype=float)
    c = np.array([cA, cB, cAB], dtype=float)
    K = np.array([Kd], dtype=float)
    N = np.array([[1, 1, -1]], dtype=float)

    c_series = pd.Series(
        {
            "[A]__0": cA0,
            "[B]__0": cB0,
            "[AB]__0": 0.0,
            "[A]": c[0],
            "[B]": c[1],
            "[AB]": c[2],
        }
    )
    N_df = pd.DataFrame(data=N, columns=["A", "B", "AB"])
    N_df["equilibrium constant"] = K

    return c, c0, N, K, c_series, N_df


def competition_binding(cA0, cB0, cC0, Kd1, Kd2):
    """
    Compute equilibrium for competition binding,
    AB <=> A + B
    AC <=> A + C
    c0 = (A, B, C)
    """
    # Coefficients for third order polynomial
    beta = Kd1 + Kd2 + cB0 + cC0 - cA0
    gamma = Kd1 * (cC0 - cA0) + Kd2 * (cB0 - cA0) + Kd1 * Kd2
    delta = -Kd1 * Kd2 * cA0

    # Compute roots, one of them is cA
    poly_roots = np.roots(np.array([1.0, beta, gamma, delta]))

    # Get index of root we want (real, between 0 and cA0)
    inds = (np.isreal(poly_roots)) & (0 < poly_roots) & (poly_roots < cA0)
    cA = poly_roots[inds][0]

    # Compute remaining concentrations
    cAC = cA * cC0 / (Kd2 + cA)
    cC = cC0 - cAC
    cAB = cA0 - cA - cAC
    cB = cB0 - cAB

    # Return problem for checking
    c0 = np.array([cA0, cB0, cC0, 0, 0], dtype=float)
    c = np.array([cA, cB, cC, cAB, cAC], dtype=float)
    K = np.array([Kd1, Kd2], dtype=float)
    N = np.array([[1, 1, 0, -1, 0], [1, 0, 1, 0, -1]], dtype=float)

    c_series = pd.Series(
        {
            "[A]__0": cA0,
            "[B]__0": cB0,
            "[C]__0": cC0,
            "[AB]__0": 0.0,
            "[AC]__0": 0.0,
            "[A]": c[0],
            "[B]": c[1],
            "[C]": c[2],
            "[AB]": c[3],
            "[AC]": c[4],
        }
    )
    N_df = pd.DataFrame(data=N, columns=["A", "B", "C", "AB", "AC"])
    N_df["equilibrium constant"] = K

    return c, c0, N, K, c_series, N_df


@hypothesis.given(
    hs.tuples(
        hs.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1.0),
        hs.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1.0),
        hs.floats(
            allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1000.0
        ),
    )
)
def test_eqcheck_simple_binding(input_tuple):
    cA0, cB0, Kd = input_tuple
    c, c0, N, K, c_series, N_df = simple_binding(cA0, cB0, Kd)

    eq_check, cons_check = eqtk.eqcheck(c, c0, N, K)
    assert eq_check.shape == (1,)
    assert cons_check.shape == (2,)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert np.allclose(cons_check, 1.0), "Error in conservation check."

    erroneous_c = 1.0001 * c
    eq_check, cons_check = eqtk.eqcheck(erroneous_c, c0, N, K)
    assert eq_check.shape == (1,)
    assert cons_check.shape == (2,)
    assert not np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert not np.allclose(cons_check, 1.0), "Error in conservation check."

    eq_check, cons_check = eqtk.eqcheck(c_series, N=N_df)
    assert eq_check.shape == (1,)
    assert cons_check.shape == (2,)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check with series."
    assert np.allclose(cons_check, 1.0), "Error in conservation check with series."

    c_series[~c_series.index.str.contains("__0")] *= 1.0001
    eq_check, cons_check = eqtk.eqcheck(c_series, N=N_df)
    assert eq_check.shape == (1,)
    assert cons_check.shape == (2,)
    assert not np.allclose(eq_check, 1.0), "Error in equilibrium check with series."
    assert not np.allclose(cons_check, 1.0), "Error in conservation check with series."


@hypothesis.given(
    hs.tuples(
        hs.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1.0),
        hs.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1.0),
        hs.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1.0),
        hs.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1.0),
        hs.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1.0),
        hs.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1.0),
        hs.floats(
            allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1000.0
        ),
    )
)
def test_eqcheck_simple_binding_multiple_inputs(input_tuple):
    cA0_0, cB0_0, cA0_1, cB0_1, cA0_2, cB0_2, Kd = input_tuple
    c_0, c0_0, N, K, c_series_0, N_df = simple_binding(cA0_0, cB0_0, Kd)
    c_1, c0_1, N, K, c_series_1, N_df = simple_binding(cA0_1, cB0_1, Kd)
    c_2, c0_2, N, K, c_series_2, N_df = simple_binding(cA0_2, cB0_2, Kd)

    c_df = pd.concat(
        (
            c_series_0.to_frame().transpose(),
            c_series_1.to_frame().transpose(),
            c_series_1.to_frame().transpose(),
        )
    )

    c = np.vstack((c_0, c_1, c_2))
    c0 = np.vstack((c0_0, c0_1, c0_2))

    eq_check, cons_check = eqtk.eqcheck(c, c0, N, K)
    assert eq_check.shape == (3, 1)
    assert cons_check.shape == (3, 2)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert np.allclose(cons_check, 1.0), "Error in conservation check."

    erroneous_c = 1.0001 * c
    eq_check, cons_check = eqtk.eqcheck(erroneous_c, c0, N, K)
    assert eq_check.shape == (3, 1)
    assert cons_check.shape == (3, 2)
    assert not np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert not np.allclose(cons_check, 1.0), "Error in conservation check."

    eq_check, cons_check = eqtk.eqcheck(c_df, N=N_df)
    assert eq_check.shape == (3, 1)
    assert cons_check.shape == (3, 2)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check with series."
    assert np.allclose(cons_check, 1.0), "Error in conservation check with series."

    c_df.loc[:, ~c_df.columns.str.contains("__0")] *= 1.0001
    eq_check, cons_check = eqtk.eqcheck(c_df, N=N_df)
    assert eq_check.shape == (3, 1)
    assert cons_check.shape == (3, 2)
    assert not np.allclose(eq_check, 1.0), "Error in equilibrium check with series."
    assert not np.allclose(cons_check, 1.0), "Error in conservation check with series."


@hypothesis.given(
    hs.tuples(
        hs.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1.0),
        hs.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1.0),
        hs.floats(allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1.0),
        hs.floats(
            allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1000.0
        ),
        hs.floats(
            allow_nan=False, allow_infinity=False, min_value=1e-6, max_value=1000.0
        ),
    )
)
def test_eqcheck_competition_binding(input_tuple):
    cA0, cB0, cC0, Kd1, Kd2 = input_tuple
    c, c0, N, K, c_series, N_df = competition_binding(cA0, cB0, cC0, Kd1, Kd2)

    eq_check, cons_check = eqtk.eqcheck(c, c0, N, K)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert np.allclose(cons_check, 1.0), "Error in conservation check."

    erroneous_c = 1.0001 * c
    eq_check, cons_check = eqtk.eqcheck(erroneous_c, c0, N, K)
    assert not np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert not np.allclose(cons_check, 1.0), "Error in conservation check."

    eq_check, cons_check = eqtk.eqcheck(c_series, N=N_df)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check with series."
    assert np.allclose(cons_check, 1.0), "Error in conservation check with series."

    c_series[~c_series.index.str.contains("__0")] *= 1.0001
    eq_check, cons_check = eqtk.eqcheck(c_series, N=N_df)
    assert not np.allclose(eq_check, 1.0), "Error in equilibrium check with series."
    assert not np.allclose(cons_check, 1.0), "Error in conservation check with series."
