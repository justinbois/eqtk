import numpy as np
import pandas as pd
import pytest

import eqtk
import eqtk.checks
import eqtk.testcases

import pandas.testing

import hypothesis
import hypothesis.strategies as hs
import hypothesis.extra.numpy as hnp


def test_c_from_df():
    # From 1-D Numpy array
    c_in = np.array([0.0, 1.0, 0.0])
    c, units, c_as_log = eqtk.checks._c_from_df(c_in, c0=False)
    assert type(c) == np.ndarray
    assert c.shape == (3,)
    assert np.array_equal(c, c_in)
    assert units == "numpy"
    assert c_as_log is None

    # From 2-D Numpy array
    c_in = np.array([[0.0, 1.0, 0.0], [0.25, 0.75, 0.0]])
    c, units, c_as_log = eqtk.checks._c_from_df(c_in, c0=False)
    assert type(c) == np.ndarray
    assert c.shape == (2, 3)
    assert np.array_equal(c, c_in)
    assert units == "numpy"
    assert c_as_log is None

    # From Pandas series
    c_in = pd.Series(
        {
            "[A]__0": 0.0,
            "[B]__0": 1.0,
            "[C]__0": 0.0,
            "[A]": 0.25,
            "[B]": 0.75,
            "[C]": 0.0,
        }
    )
    target_c0 = pd.Series({"A": 0.0, "B": 1.0, "C": 0.0})
    target_c = pd.Series({"A": 0.25, "B": 0.75, "C": 0.0})
    c, units, c_as_log = eqtk.checks._c_from_df(c_in, c0=False)
    assert type(c) == pd.core.series.Series
    assert units == None
    assert c_as_log == False
    pandas.testing.assert_series_equal(c, target_c)
    pandas.testing.assert_series_equal(c, target_c)

    c0, units, c0_as_log = eqtk.checks._c_from_df(c_in, c0=True)
    assert type(c0) == pd.core.series.Series
    assert units == None
    assert c0_as_log == False
    pandas.testing.assert_series_equal(c0, target_c0)
    pandas.testing.assert_series_equal(c0, target_c0)

    # From Pandas series with units
    c_in = pd.Series(
        {
            "[A]__0 (mM)": 0.0,
            "[B]__0 (mM)": 1.0,
            "[C]__0 (mM)": 0.0,
            "[A] (mM)": 0.25,
            "[B] (mM)": 0.75,
            "[C] (mM)": 0.0,
        }
    )
    target_c0 = pd.Series({"A": 0.0, "B": 1.0, "C": 0.0})
    target_c = pd.Series({"A": 0.25, "B": 0.75, "C": 0.0})
    c, units, c_as_log = eqtk.checks._c_from_df(c_in, c0=False)
    assert type(c) == pd.core.series.Series
    assert units == "mM"
    assert c_as_log == False
    pandas.testing.assert_series_equal(c, target_c)
    pandas.testing.assert_series_equal(c, target_c)

    c0, units, c0_as_log = eqtk.checks._c_from_df(c_in, c0=True)
    assert type(c0) == pd.core.series.Series
    assert units == "mM"
    assert c0_as_log == False
    pandas.testing.assert_series_equal(c0, target_c0)
    pandas.testing.assert_series_equal(c0, target_c0)

    # From Pandas series with logaithrms
    c_in = pd.Series(
        {
            "[A]__0": 0.0,
            "[B]__0": 1.0,
            "[C]__0": 0.0,
            "ln [A]": -2.3,
            "ln [B]": -1.4,
            "ln [C]": -np.inf,
        }
    )
    target_c0 = pd.Series({"A": 0.0, "B": 1.0, "C": 0.0})
    target_c = pd.Series({"A": -2.3, "B": -1.4, "C": -np.inf})
    c, units, c_as_log = eqtk.checks._c_from_df(c_in, c0=False)
    assert type(c) == pd.core.series.Series
    assert units == None
    assert c_as_log == True
    pandas.testing.assert_series_equal(c, target_c)
    pandas.testing.assert_series_equal(c, target_c)

    c0, units, c0_as_log = eqtk.checks._c_from_df(c_in, c0=True)
    assert type(c0) == pd.core.series.Series
    assert units == None
    assert c0_as_log == False
    pandas.testing.assert_series_equal(c0, target_c0)
    pandas.testing.assert_series_equal(c0, target_c0)


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
    tc = eqtk.testcases.simple_binding(cA0, cB0, Kd)

    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        tc["c"], tc["c0"], tc["N"], tc["K"], return_detailed=True
    )
    assert all_ok
    assert eq_check.shape == (1,)
    assert cons_check.shape == (2,)
    assert eq_satisfied.shape == (1,)
    assert cons_satisfied.shape == (2,)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert np.all(eq_satisfied), "Error in conservation check."
    assert np.all(cons_satisfied), "Error in conservation check."
    assert eqtk.eqcheck(tc["c"], tc["c0"], tc["N"], tc["K"])

    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        tc["c_log"], tc["c0"], tc["N"], tc["K"], c_as_log=True, return_detailed=True
    )
    assert all_ok
    assert eq_check.shape == (1,)
    assert cons_check.shape == (2,)
    assert eq_satisfied.shape == (1,)
    assert cons_satisfied.shape == (2,)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert np.all(eq_satisfied), "Error in conservation check."
    assert np.all(cons_satisfied), "Error in conservation check."
    assert eqtk.eqcheck(tc["c_log"], tc["c0"], tc["N"], tc["K"], c_as_log=True)

    tc["erroneous_c"] = 1.0001 * tc["c"]
    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        tc["erroneous_c"], tc["c0"], tc["N"], tc["K"], return_detailed=True
    )
    assert not all_ok
    assert eq_check.shape == (1,)
    assert cons_check.shape == (2,)
    assert eq_satisfied.shape == (1,)
    assert cons_satisfied.shape == (2,)
    assert not np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert not np.all(eq_satisfied), "Error in conservation check."
    assert not np.all(cons_satisfied), "Error in conservation check."
    assert not eqtk.eqcheck(tc["erroneous_c"], tc["c0"], tc["N"], tc["K"])

    tc["erroneous_c_log"] = tc["c_log"] - 0.001
    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        tc["erroneous_c_log"],
        tc["c0"],
        tc["N"],
        tc["K"],
        c_as_log=True,
        return_detailed=True,
    )
    assert not all_ok
    assert eq_check.shape == (1,)
    assert cons_check.shape == (2,)
    assert eq_satisfied.shape == (1,)
    assert cons_satisfied.shape == (2,)
    assert not np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert not np.all(eq_satisfied), "Error in conservation check."
    assert not np.all(cons_satisfied), "Error in conservation check."
    assert not eqtk.eqcheck(
        tc["erroneous_c_log"], tc["c0"], tc["N"], tc["K"], c_as_log=True
    )

    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        c=tc["c_series"], N=tc["N_df"], return_detailed=True
    )
    assert all_ok
    assert eq_check.shape == (1,)
    assert cons_check.shape == (2,)
    assert eq_satisfied.shape == (1,)
    assert cons_satisfied.shape == (2,)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert np.all(eq_satisfied), "Error in conservation check."
    assert np.all(cons_satisfied), "Error in conservation check."
    assert eqtk.eqcheck(c=tc["c_series"], N=tc["N_df"])

    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        c=tc["c_series_log"], N=tc["N_df"], return_detailed=True
    )
    assert all_ok
    assert eq_check.shape == (1,)
    assert cons_check.shape == (2,)
    assert eq_satisfied.shape == (1,)
    assert cons_satisfied.shape == (2,)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert np.all(eq_satisfied), "Error in conservation check."
    assert np.all(cons_satisfied), "Error in conservation check."
    assert eqtk.eqcheck(c=tc["c_series_log"], N=tc["N_df"])

    tc["c_series"][~tc["c_series"].index.str.contains("__0")] *= 1.0001
    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        tc["c_series"], N=tc["N_df"], return_detailed=True
    )
    assert not all_ok
    assert eq_check.shape == (1,)
    assert cons_check.shape == (2,)
    assert eq_satisfied.shape == (1,)
    assert cons_satisfied.shape == (2,)
    assert not np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert not np.all(eq_satisfied), "Error in conservation check."
    assert not np.all(cons_satisfied), "Error in conservation check."
    assert not eqtk.eqcheck(c=tc["c_series"], N=tc["N_df"])


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

    tc_0 = eqtk.testcases.simple_binding(cA0_0, cB0_0, Kd)
    tc_1 = eqtk.testcases.simple_binding(cA0_0, cB0_0, Kd)
    tc_2 = eqtk.testcases.simple_binding(cA0_0, cB0_0, Kd)

    c_df = pd.concat(
        (
            tc_0["c_series"].to_frame().transpose(),
            tc_1["c_series"].to_frame().transpose(),
            tc_2["c_series"].to_frame().transpose(),
        )
    )

    c_df_log = pd.concat(
        (
            tc_0["c_series_log"].to_frame().transpose(),
            tc_1["c_series_log"].to_frame().transpose(),
            tc_2["c_series_log"].to_frame().transpose(),
        )
    )

    c = np.vstack((tc_0["c"], tc_1["c"], tc_2["c"]))
    c0 = np.vstack((tc_0["c0"], tc_1["c0"], tc_2["c0"]))

    N = tc_0["N"]
    K = tc_0["K"]
    N_df = tc_0["N_df"]

    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        c, c0, N, K, return_detailed=True
    )
    assert all_ok
    assert eq_check.shape == (3, 1)
    assert cons_check.shape == (3, 2)
    assert eq_satisfied.shape == (3, 1)
    assert cons_satisfied.shape == (3, 2)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert np.all(eq_satisfied)
    assert np.all(cons_satisfied)
    assert eqtk.eqcheck(c, c0, N, K)

    erroneous_c = 1.0001 * c
    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        erroneous_c, c0, N, K, return_detailed=True
    )
    assert not all_ok
    assert eq_check.shape == (3, 1)
    assert cons_check.shape == (3, 2)
    assert eq_satisfied.shape == (3, 1)
    assert cons_satisfied.shape == (3, 2)
    assert not np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert not np.all(eq_satisfied)
    assert not np.all(cons_satisfied)
    assert not eqtk.eqcheck(erroneous_c, c0, N, K)

    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        c=c_df, N=N_df, return_detailed=True
    )
    assert all_ok
    assert eq_check.shape == (3, 1)
    assert cons_check.shape == (3, 2)
    assert eq_satisfied.shape == (3, 1)
    assert cons_satisfied.shape == (3, 2)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert np.all(eq_satisfied)
    assert np.all(cons_satisfied)
    assert eqtk.eqcheck(c=c_df, N=N_df)

    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        c_df_log, N=N_df, return_detailed=True
    )
    assert all_ok
    assert eq_check.shape == (3, 1)
    assert cons_check.shape == (3, 2)
    assert eq_satisfied.shape == (3, 1)
    assert cons_satisfied.shape == (3, 2)
    assert np.allclose(eq_check, 1.0), "Error in equilibrium check with series."
    assert np.all(eq_satisfied), "Error in equilibrium check with series."
    assert np.all(cons_satisfied), "Error in conservation check with series."
    assert eqtk.eqcheck(c_df_log, N=N_df)

    c_df.loc[:, ~c_df.columns.str.contains("__0")] *= 1.0001
    all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied = eqtk.eqcheck(
        c_df, N=N_df, return_detailed=True
    )
    assert not all_ok
    assert eq_check.shape == (3, 1)
    assert cons_check.shape == (3, 2)
    assert eq_satisfied.shape == (3, 1)
    assert cons_satisfied.shape == (3, 2)
    assert not np.allclose(eq_check, 1.0), "Error in equilibrium check."
    assert not np.all(eq_satisfied)
    assert not np.all(cons_satisfied)
    assert not eqtk.eqcheck(c=c_df, N=N_df)
