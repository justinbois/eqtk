import numpy as np
import pandas as pd

from . import solvers
from . import linalg
from . import parsers
from . import constants


def _c_from_df(c, c0=False):
    """Extract concentrations from outputted data frame, or return c if
    if is a Numpy array.
    """
    if type(c) not in [pd.core.series.Series, pd.core.frame.DataFrame]:
        return c, "numpy", None

    if type(c) == pd.core.series.Series:
        if c0:
            cols = c.index[c.index.str.contains("]__0")]
        else:
            cols = c.index[~c.index.str.contains("]__0")]
    elif type(c) == pd.core.frame.DataFrame:
        if c0:
            cols = c.columns[c.columns.str.contains("]__0")]
        else:
            cols = c.columns[~c.columns.str.contains("]__0")]

    df = c[cols]

    c_as_log = cols[0][:3] == "ln "

    # Rename columns to just be chemical species
    if c_as_log:
        names = {col: col[4 : col.find("]")] for col in cols}
    else:
        names = {col: col[1 : col.find("]")] for col in cols}
    if type(c) == pd.core.series.Series:
        df = df.rename(index=names)
    if type(c) == pd.core.frame.DataFrame:
        df = df.rename(columns=names)

    # Determine units
    col = cols[0]
    if c0:
        if col[-3:] == "__0":
            units = None
        else:
            units = col[col.rfind("(") + 1 : -1]
    else:
        if col[-1] == "]":
            units = None
        else:
            units = col[col.rfind("(") + 1 : -1]

    return df, units, c_as_log


def eqcheck(
    c,
    c0=None,
    N=None,
    K=None,
    logK=None,
    A=None,
    G=None,
    names=None,
    units="unspecified",
    G_units=None,
    solvent_density=None,
    T=293.15,
    c_as_log="unspecified",
    normal_A=True,
    eq_rtol=1e-5,
    eq_atol=1e-8,
    tol=1e-7,
    tol_zero=1e-12,
    return_detailed=False,
):
    """Check the satisfaction of equilibrium expressions and
    conservation laws.

    Parameters
    ----------
    c : array or DataFrame, shape c0.shape
        Equilibrium concentrations of all species. `c[i, j]` is the
        equilibrium concentration of species `j` for initial
        concentrations given by `c0[i, :]` in units given by `units`.
    c0 : array_like, dict, Series, or DataFrame, shape (n_points, n_compounds) or (n_compounds, ), or None
        Each row contains the total "initial" concentration of all
        possible chemical species in solution. The equilibrium
        concentration of all species is computed for each row in `c0`.
        `c0[i, j]` is the initial concentration of compound `j` for
        calculation `i`. `c0` may also be passed as a Pandas Series
        where the indices contain the name of the chemical species and
        each value is the "initial concentration." `c0` may also be
        passed as a Pandas DataFrame where each row contains the total
        "initial" concentration of all possible compounds in solution
        and the column names are the names of the chemical species. If
        `c0` is passed as a dict, the dict must be convertible to a
        Pandas Series or DataFrame as `pd.Series(c0)` or
        `pd.DataFrame(c0)`. If `c` is inputted as a Series or DataFrame
        and has initial concentrations, `c0` may be None.
    N : array_like or DataFrame, default `None`
        Stoichiometic matrix. `N[r, j]` = the stoichiometric coefficient
        of compound `j` in chemical reaction `r`. All rows of `N` must
        be linearly independent. If entered as a DataFrame, the name of
        chemical species `j` is `N.columns[j]`. Optionally, column
        `'equilibrium constant'` contains the equilibrium constants for
        each reaction in units commensurate with those of `c0`. If `N`
        is given, `A` and `G` cannot be given.
    K : array_like, shape (n_reactions,), default `None`
        `K[r]` is the equilibrium constant for chemical reaction r in
        units commensurate with those of `c0`. If `N` is given as a
        DataFrame with an `'equilibrium constant'` column, `K` should
        not be supplied. If `K`is given, `A` and `G` cannot be given.
    logK : array_like, shape (n_reactions,), default `None`
        `logK[r]` is the natural logarithm of the equilibrium constant
        for chemical reaction r. If `logK` is specified, the
        concentrations must all be dimensionless (`units=None`). If `N`
        is given as a DataFrame with a `'log equilibrium constant'`
        column, `logK` should not be supplied. If `K` is given, `A`,
        `G`, and `K` cannot be given.
    A : array_like or DataFrame, n_compounds columns
        Conservation matrix. If `c` is the output, then
        `A @ c0 = A @ c`. All entries must be nonnegative and the rows
        of `A` must be linearly independent. If entered as a DataFrame,
        the name of chemical species `j` is `A.columns[j]`. If `A` is
        given, `G` must be given, and `N` and `K` cannot be given.
    G : array_like, shape (n_compounds, ), default `None`
        `G[j]` is the free energy of chemical species `j` in units
        specified by `G_units`. If `G` is given, `A` must be given, and
        `N` and `K` cannot be given.
    names : list or tuple of str, default `None`, optional
        The names of the chemical species. Names are inferred if `N` or
        `A` is given as a DataFrame, in which case `names` is
        unnecessary.
    units : string or `None`, default 'unspecified'
        The units of the concentrations. Allowable values are {`None`,
        'mole fraction', 'molar', 'M', 'millimolar', 'mM', 'micromolar',
        'uM', 'µM', 'nanomolar', 'nM', 'picomolar', 'pM'}. If
        'unspecified', units are inferred from other inputs. If the `c`
        is a Numpy array, units are considered `None` if 'unspecified'.
        The equilibrium constants given by `K` must have corresponding
        units.
    G_units : string, default `None`
        Units in which free energy is given. If `None` or `'kT'`, the
        free  energies are specified in units of of the thermal energy
        kT. Allowable values are {None, 'kT', kcal/mol', 'J', 'J/mol',
        'kJ/mol', 'pN-nm'}.
    solvent_density : float, default `None`
        The density of the solvent in units commensurate with the
        `units` keyword argument. Default (`None`) assumes the solvent
        is water, and its density is computed at the temperature
        specified by the `T` keyword argument.
    T : float, default = 293.15
        Temperature, in Kelvin, of the solution. When `N` and `K` are
        given, `T` is ignored if `solvent_density` is given or if
        `units` is `None`. If `A` and `G` are given, `T` is ignored when
        `units` and `G_units` are both `None`.
    c_as_log : str or bool, default 'unspecified'
        If True, `c` is given with computed concentrations as the
        natural logarithm of their value, as would be returned by
        `eqtk.solve()` with the `return_log=True` keyword argument. If
        'unspecified', logarithmic `c` is inferred from the input. If
        `c` is a Numpy array and `c_as_log` is 'unspecified', the
        inputted `c` is assumed to be concentrations and not their
        natural logarithms.
    normal_A : bool, default True
        If True, perform a transformation on `A` such that its rows are
        normal vectors when checking conservation laws. If False, use
        inputted `A` directly as the conservation matrix. This is
        ignored if `A` is not specified; the resulting conservation
        matrix in that case has orthonormal rows by construction. This
        parameter should be chosen to match what was used in the
        calculation being checked because the convergence criteria
        change slightly depending on the particular values in `A`.
    eq_atol : float, default 1e-8
        Absolute tolerance parameter to check closeness of equilibrium
        as used in `numpy.isclose()`.
    eq_rtol : float, default 1e-5
        Relative tolerance parameter to check closeness of equilibrium
        condition as used in `numpy.isclose()`.
    tol : float, default 1e-7
        Tolerance for conservation law. The absolute tolerance for a 
        given initial concentration c0 (a one-dimensional array) for the
        conservation conditions are tol * np.dot(A, c0) except when an 
        entry in np.dot(A, c0) is zero. If that is the case for entry i, 
        then the absolute tolerance is tol * np.max(A[i] * c0). If all 
        entries in A[i] * c0 are zero, which only happens with c0 = 0, 
        the absolute tolerance is `tol_zero`.
    tol_zero : float, default 1e-12
        Absolute tolerance for convergence when the initial 
        concentrations are all zero. This cannot really be set a priori; 
        it depends on the scale of the dimensionless concentrations. By 
        default, assume an absolute tolerance consistent with `tol` and
        millimolar mole fractions.
    return_detailed : bool, default False
        If True, return detailed quantitative checks (see return values
        below). If False, only a boolean is returned, True if all 
        equilibrium and conservation conditions are met and False 
        otherwise.

    Returns
    -------
    output : bool
        True is all equilibrium and conservation conditions are met.
        False otherwise.
    equilibrium_check : Numpy array, shape (n_points, n_reactions) or (n_reactions, )
        equilibrium_check[i, r] is the ratio of the left and right hand
        sides of the mass action expression for equilibrium for
        calculation `i`.  This is defined as
        (prod_j c[j]**N[r][j]) / K[r]. If `N` or `K` is not
        given, they are calculated from `A` and `G`. This is only 
        returned if `return_detailed` is True.
    equilibrium_satisfied : Numpy array, shape (n_points, n_reactions) or (n_reactions, )
        equilibrium_satisfied[i, r] is True if the equilibrium condition
        for reaction r in calculation i is satisfied.
        is met and False otherwise. This is only  returned if 
        `return_detailed` is True.
    cons_check : Numpy array, shape (n_points, n_compounds - n_reactions), or (n_compounds - n_reactions,)
        cons_check[i, k] is the difference between the calculated value 
        of a conserved quantity and the actual value for calculation
        `i`. It is defined as cons_check = np.dot(A, c) - np.dot(A, c0).
        If `A` is not given, `A` is calculated as the null space of `N`.
        This is only returned if `return_detailed` is True.
    cons_satisfied : Numpy array, shape (n_points, n_reactions) or (n_reactions, )
        cons_satisfied[i, k] is True if the conservation condition k for
        calculation i is met and False otherwise. This is only  returned 
        if `return_detailed` is True.

    """
    eq_check, eq_satisfied, cons_check, cons_satisfied = _eqcheck_quant(
        c,
        c0=c0,
        N=N,
        K=K,
        logK=logK,
        A=A,
        G=G,
        names=names,
        units=units,
        G_units=G_units,
        solvent_density=solvent_density,
        T=T,
        c_as_log=c_as_log,
        normal_A=normal_A,
        eq_rtol=eq_rtol,
        eq_atol=eq_atol,
        tol=tol,
        tol_zero=tol_zero,
    )

    all_ok = eq_satisfied.all() and cons_satisfied.all()

    if return_detailed:
        return all_ok, eq_check, eq_satisfied, cons_check, cons_satisfied

    return all_ok


def _eqcheck_quant(
    c,
    c0=None,
    N=None,
    K=None,
    logK=None,
    A=None,
    G=None,
    names=None,
    units="unspecified",
    G_units=None,
    solvent_density=None,
    T=293.15,
    c_as_log="unspecified",
    normal_A=True,
    eq_rtol=1e-5,
    eq_atol=1e-8,
    tol=1e-7,
    tol_zero=1e-12,
):
    """Compute the error in satisfaction of equilibrium expressions
    and in conservation laws.

    Parameters
    ----------
    c : array or DataFrame, shape c0.shape
        Equilibrium concentrations of all species. `c[i, j]` is the
        equilibrium concentration of species `j` for initial
        concentrations given by `c0[i, :]` in units given by `units`.
    c0 : array_like, dict, Series, or DataFrame, shape (n_points, n_compounds) or (n_compounds, ), or None
        Each row contains the total "initial" concentration of all
        possible chemical species in solution. The equilibrium
        concentration of all species is computed for each row in `c0`.
        `c0[i, j]` is the initial concentration of compound `j` for
        calculation `i`. `c0` may also be passed as a Pandas Series
        where the indices contain the name of the chemical species and
        each value is the "initial concentration." `c0` may also be
        passed as a Pandas DataFrame where each row contains the total
        "initial" concentration of all possible compounds in solution
        and the column names are the names of the chemical species. If
        `c0` is passed as a dict, the dict must be convertible to a
        Pandas Series or DataFrame as `pd.Series(c0)` or
        `pd.DataFrame(c0)`. If `c` is inputted as a Series or DataFrame
        and has initial concentrations, `c0` may be None.
    N : array_like or DataFrame, default `None`
        Stoichiometic matrix. `N[r, j]` = the stoichiometric coefficient
        of compound `j` in chemical reaction `r`. All rows of `N` must
        be linearly independent. If entered as a DataFrame, the name of
        chemical species `j` is `N.columns[j]`. Optionally, column
        `'equilibrium constant'` contains the equilibrium constants for
        each reaction in units commensurate with those of `c0`. If `N`
        is given, `A` and `G` cannot be given.
    K : array_like, shape (n_reactions,), default `None`
        `K[r]` is the equilibrium constant for chemical reaction r in
        units commensurate with those of `c0`. If `N` is given as a
        DataFrame with an `'equilibrium constant'` column, `K` should
        not be supplied. If `K`is given, `A` and `G` cannot be given.
    logK : array_like, shape (n_reactions,), default `None`
        `logK[r]` is the natural logarithm of the equilibrium constant
        for chemical reaction r. If `logK` is specified, the
        concentrations must all be dimensionless (`units=None`). If `N`
        is given as a DataFrame with a `'log equilibrium constant'`
        column, `logK` should not be supplied. If `K` is given, `A`,
        `G`, and `K` cannot be given.
    A : array_like or DataFrame, n_compounds columns
        Conservation matrix. If `c` is the output, then
        `A @ c0 = A @ c`. All entries must be nonnegative and the rows
        of `A` must be linearly independent. If entered as a DataFrame,
        the name of chemical species `j` is `A.columns[j]`. If `A` is
        given, `G` must be given, and `N` and `K` cannot be given.
    G : array_like, shape (n_compounds, ), default `None`
        `G[j]` is the free energy of chemical species `j` in units
        specified by `G_units`. If `G` is given, `A` must be given, and
        `N` and `K` cannot be given.
    names : list or tuple of str, default `None`, optional
        The names of the chemical species. Names are inferred if `N` or
        `A` is given as a DataFrame, in which case `names` is
        unnecessary.
    units : string or `None`, default 'unspecified'
        The units of the concentrations. Allowable values are {`None`,
        'mole fraction', 'molar', 'M', 'millimolar', 'mM', 'micromolar',
        'uM', 'µM', 'nanomolar', 'nM', 'picomolar', 'pM'}. If
        'unspecified', units are inferred from other inputs. If the `c`
        is a Numpy array, units are considered `None` if 'unspecified'.
        The equilibrium constants given by `K` must have corresponding
        units.
    G_units : string, default `None`
        Units in which free energy is given. If `None` or `'kT'`, the
        free  energies are specified in units of of the thermal energy
        kT. Allowable values are {None, 'kT', kcal/mol', 'J', 'J/mol',
        'kJ/mol', 'pN-nm'}.
    solvent_density : float, default `None`
        The density of the solvent in units commensurate with the
        `units` keyword argument. Default (`None`) assumes the solvent
        is water, and its density is computed at the temperature
        specified by the `T` keyword argument.
    T : float, default = 293.15
        Temperature, in Kelvin, of the solution. When `N` and `K` are
        given, `T` is ignored if `solvent_density` is given or if
        `units` is `None`. If `A` and `G` are given, `T` is ignored when
        `units` and `G_units` are both `None`.
    c_as_log : str or bool, default 'unspecified'
        If True, `c` is given with computed concentrations as the
        natural logarithm of their value, as would be returned by
        `eqtk.solve()` with the `return_log=True` keyword argument. If
        'unspecified', logarithmic `c` is inferred from the input. If
        `c` is a Numpy array and `c_as_log` is 'unspecified', the
        inputted `c` is assumed to be concentrations and not their
        natural logarithms.
    normal_A : bool, default True
        If True, perform a transformation on `A` such that its rows are
        normal vectors when checking conservation laws. If False, use
        inputted `A` directly as the conservation matrix. This is
        ignored if `A` is not specified; the resulting conservation
        matrix in that case has orthonormal rows by construction. This
        parameter should be chosen to match what was used in the
        calculation being checked because the convergence criteria
        change slightly depending on the particular values in `A`.
    eq_atol : float, default 1e-8
        Absolute tolerance parameter to check closeness of equilibrium
        as used in `numpy.isclose()`.
    eq_rtol : float, default 1e-5
        Relative tolerance parameter to check closeness of equilibrium
        condition as used in `numpy.isclose()`.
    tol : float, default 1e-7
        Tolerance for conservation law. The absolute tolerance for a 
        given initial concentration c0 (a one-dimensional array) for the
        conservation conditions are tol * np.dot(A, c0) except when an 
        entry in np.dot(A, c0) is zero. If that is the case for entry i, 
        then the absolute tolerance is tol * np.max(A[i] * c0). If all 
        entries in A[i] * c0 are zero, which only happens with c0 = 0, 
        the absolute tolerance is `tol_zero`.
    tol_zero : float, default 1e-12
        Absolute tolerance for convergence when the initial 
        concentrations are all zero. This cannot really be set a priori; 
        it depends on the scale of the dimensionless concentrations. By 
        default, assume an absolute tolerance consistent with `tol` and
        millimolar mole fractions.

    Returns
    -------
    equilibrium_check : Numpy array, shape (n_points, n_reactions) or (n_reactions, )
        equilibrium_check[i, r] is the ratio of the left and right hand
        sides of the mass action expression for equilibrium for
        calculation `i`.  This is defined as
        (prod_j c[j]**N[r][j]) / K[r]. If `N` or `K` is not
        given, they are calculated from `A` and `G`.
    equilibrium_satisfied : Numpy array, shape (n_points, n_reactions) or (n_reactions, )
        equilibrium_satisfied[i, r] is True if the equilibrium condition
        for reaction r in calculation i is satisfied.
        is met and False otherwise.
    cons_check : Numpy array, shape (n_points, n_compounds - n_reactions), or (n_compounds - n_reactions,)
        cons_check[i, k] is the difference between the calculated value 
        of a conserved quantity and the actual value for calculation
        `i`. It is defined as cons_check = np.dot(A, c) - np.dot(A, c0).
        If `A` is not given, `A` is calculated as the null space of `N`.
    cons_satisfied : Numpy array, shape (n_points, n_reactions) or (n_reactions, )
        cons_satisfied[i, k] is True if the conservation condition k for
        calculation i is met and False otherwise.


    """
    if units == "unspecified" and type(c) not in (
        pd.core.series.Series,
        pd.core.frame.DataFrame,
    ):
        units = None

    if c_as_log == "unspecified" and type(c) not in (
        pd.core.series.Series,
        pd.core.frame.DataFrame,
    ):
        c_as_log = None

    if c0 is None:
        if type(c) not in (pd.core.series.Series, pd.core.frame.DataFrame):
            raise ValueError(
                "If `c0` is not specified, `c` must be provided as a Series or DataFrame that contains the initial concentrations."
            )
        c0, _, _ = _c_from_df(c, c0=True)

    c, inferred_units, c_as_log_inferred = _c_from_df(c, c0=False)

    if units == "unspecified":
        units = inferred_units

    if inferred_units != "numpy" and inferred_units != units:
        raise ValueError("Units in inputted `c` do not match `units` keyword argument.")

    x0, N, logK, A, G, names, solvent_density, single_point = parsers.parse_input(
        c0=c0,
        N=N,
        K=K,
        logK=logK,
        A=A,
        G=G,
        names=names,
        units=units,
        solvent_density=solvent_density,
        T=T,
        G_units=G_units,
    )

    if c_as_log == "unspecified":
        c_as_log = c_as_log_inferred
    elif c_as_log == True and c_as_log_inferred == False:
        raise ValueError(
            "Inputted `c` does not have logarithmic concentrations, but `c_as_log` is specified as `True`."
        )
    elif c_as_log == False and c_as_log_inferred == True:
        raise ValueError(
            "Inputted `c` has logarithmic concentrations, but `c_as_log` is specified as `False`."
        )

    if c_as_log and solvent_density != 1.0:
        raise ValueError("If `c_as_log` is True, must have `solvent_density = 1`.")

    if type(c) in (pd.core.series.Series, pd.core.frame.DataFrame):
        x = c[names].to_numpy(dtype=float, copy=True) / solvent_density
    else:
        x = c / solvent_density

    if len(x.shape) == 1:
        x = x.reshape((1, len(x)))

    if x0.shape != x.shape:
        raise ValueError("`c0` and `c` must have the same shape.")

    if N is None:
        N = linalg.nullspace_svd(A, tol=constants._nullspace_tol)
        prune = "A"

    if A is None:
        if len(N) == 0:
            A = np.array([[1.0] + [0.0] * (N.shape[1] - 1)])
        else:
            A = linalg.nullspace_svd(N, tol=constants._nullspace_tol)
        prune = "N"

    if G is not None:
        logK = -np.dot(N, G)

    return _check_equilibrium(
        x0,
        x,
        N,
        A,
        logK,
        G,
        prune,
        single_point,
        c_as_log,
        normal_A,
        eq_atol,
        eq_rtol,
        tol,
        tol_zero,
    )


def _check_equilibrium(
    c0,
    c,
    N,
    A,
    logK,
    G,
    prune,
    single_point,
    c_as_log=False,
    normal_A=True,
    eq_atol=1e-5,
    eq_rtol=1e-8,
    tol=1e-7,
    tol_zero=1e-12,
):
    """Check to make sure equilibrium is satisfied."""
    eq_check = np.empty((c.shape[0], N.shape[0]))
    cons_check = np.empty((c.shape[0], N.shape[1] - N.shape[0]))
    eq_satisfied = np.empty((c.shape[0], N.shape[0]), dtype=bool)
    cons_satisfied = np.empty((c.shape[0], N.shape[1] - N.shape[0]), dtype=bool)

    for i, (c0_i, c_i) in enumerate(zip(c0, c)):
        eq_check_i, eq_satisfied_i, cons_check_i, cons_satisfied_i = _check_equilibrium_single_point(
            c0_i,
            c_i,
            N,
            logK,
            A,
            G,
            prune,
            c_as_log,
            normal_A,
            eq_atol,
            eq_rtol,
            tol,
            tol_zero,
        )
        eq_check[i] = eq_check_i
        eq_satisfied[i] = eq_satisfied_i
        cons_check[i] = cons_check_i
        cons_satisfied[i] = cons_satisfied_i

    if single_point:
        eq_check = eq_check.flatten()
        eq_satisfied = eq_satisfied.flatten()
        cons_check = cons_check.flatten()
        cons_satisfied = cons_satisfied.flatten()

    return eq_check, eq_satisfied, cons_check, cons_satisfied


def _check_equilibrium_single_point(
    c0,
    c,
    N,
    logK,
    A,
    G,
    prune,
    c_as_log=False,
    normal_A=True,
    eq_atol=1e-5,
    eq_rtol=1e-8,
    tol=1e-7,
    tol_zero=1e-12,
):
    """
    Check concentrations to verify equilibrium conditions are met.

    Parameters
    ----------
    c0 : array_like, shape (n_compounds,)
        c0[j] is the initial concentration of compound j.
    c : array_like, shape (n_compounds,)
        c[j] is the equilibrium concentration of compound j.
    N : array_like, shape (n_compounds - n_particles, n_compounds)
        N[r][j] = the stoichiometric coefficient of compounds j
        in chemical reaction r.
    K : array_like, shape (n_compounds - n_particles,)
        K[r] is the equilibrium constant for chemical reaction r
    A : Numpy array, shape ()

    Returns
    -------
    equilibrium_check : array_like, shape(n_compounds - n_particles,)
        equilibrium_check[r] is the fractional error in the
        equilibrium expression for chemical reaction r.  This is
        defined as (prod_j c[j]**N[r][j] - K[r]) / K[r]
    cons_check : array_like, shape(n_particles,)
        cons_check[i] is the error in conservation of mass for
        irreducible species (particle) i.  It is defined as
        cons_check = (np.dot(A, c) - c0) / c0

    Examples
    --------
    1) Find the equilibrium concentrations of a solution containing
       species A, B, C, AB, BB, and BC that can undergo chemical
       reactions

                A <--> C,      K = 50 (dimensionless)
            A + C <--> AB      K = 10 (1/mM)
            B + B <--> BB      K = 40 (1/mM)
            B + C <--> BC      K = 100 (1/mM)
       with initial concentrations of [A]_0 = 1.0 mM, [B]_0 = 3.0 mM.
       Then verify that the standard equilibrium expressions hold.

    >>> import numpy as np
    >>> import eqtk
    >>> A = np.array([[ 1,  0,  1,  1,  0,  1],
                      [ 0,  1,  0,  1,  2,  1]])
    >>> N = np.array([[-1,  0,  1,  0,  0,  0],
                      [-1, -1,  0,  1,  0,  0],
                      [ 0, -2,  0,  0,  1,  0],
                      [ 0, -1, -1,  0,  0,  1]])
    >>> K = np.array([50.0, 10.0, 40.0, 100.0])
    >>> c0 = np.array([1.0, 3.0])
    >>> c = eqtk.conc(c0, N, K, units='mM')
    >>> equilibrium_check, cons_check = eqtk.check_equilibrium(c, A, N, K, c0)
    >>> equilibrium_check
    array([  3.33066907e-16,  -4.66293670e-15,  -4.44089210e-16,
             0.00000000e+00])
    >>> cons_check
    array([ -4.11757295e-11,  -1.28775509e-11])
    """
    if prune == "N":
        _, _, _, active_compounds, active_reactions = solvers._prune_NK(N, -logK, c0)
    elif prune == "A":
        # Prune and make new stoich matrix
        A_new, G_new, _, active_compounds = solvers._prune_AG(A, G, c0)
        N_new = linalg.nullspace_svd(A_new, tol=constants._nullspace_tol)
        N = np.zeros_like(N)
        N[: len(N_new), active_compounds] = N_new
        logK = -np.dot(N, G)
        active_reactions = [True] * len(N_new) + [False] * (len(N) - len(N_new))

    # Check equilibrium expressions
    eq_check = np.empty(len(N))
    eq_satisfied = np.empty(len(N), dtype=bool)
    if c_as_log:
        logc = c
    else:
        logc = np.array([np.log(ci) for ac, ci in zip(active_compounds, c) if ac])

    for r, logK_val in enumerate(logK):
        if active_reactions[r]:
            eq_check[r] = np.exp(np.dot(N[r, active_compounds], logc) - logK_val)
        else:
            eq_check[r] = 1.0

        eq_satisfied[r] = np.isclose(eq_check[r], 1.0, atol=eq_atol, rtol=eq_rtol)

    # Check conservation expressions
    if normal_A:
        A = linalg._normalize_rows(A)

    abs_tol = solvers._tolerance(tol, tol_zero, A, c0)
    target = np.dot(A, c0)
    if c_as_log:
        c = np.exp(c)
    res = np.dot(A, c)
    cons_check = np.empty(len(res))
    cons_satisfied = np.empty(len(res), dtype=bool)
    for i, (res_val, target_val) in enumerate(zip(res, target)):
        cons_check[i] = res_val - target_val
        cons_satisfied[i] = np.abs(cons_check[i]) <= abs_tol[i]

    return eq_check, eq_satisfied, cons_check, cons_satisfied
