import numpy as np
import pandas as pd

from . import parsers
from . import trust_region
from . import linalg
from . import numba_check
from . import constants

have_numba, jit = numba_check.numba_check()


def solve(
    c0,
    N=None,
    K=None,
    A=None,
    G=None,
    names=None,
    units=None,
    G_units=None,
    solvent_density=None,
    T=293.15,
    max_iters=1000,
    tol=0.0000001,
    delta_bar=1000.0,
    eta=0.125,
    min_delta=1.0e-12,
    max_trials=100,
    perturb_scale=100.0,
):
    """Solve for equilibrium concentrations of all species in a dilute
    solution.

    Parameters
    ----------
    c0 : array_like, dict, Series, or DataFrame,
         shape (n_points, n_compounds) or (n_compounds, )
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
        `pd.DataFrame(c0)`.
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
    A : array_like or DataFrame, n_compounds columns
        Constraint matrix. If `c` is the output, then `A @ c0 = A @ c`.
        All entries must be nonnegative and the rows of `A` must be
        linearly independent. If entered as a DataFrame, the name of
        chemical species `j` is `A.columns[j]`. If `A` is given, `G`
        must be given, and `N` and `K` cannot be given.
    G : array_like, shape (n_compounds, ), default `None`
        `G[j]` is the free energy of chemical species `j` in units
        specified by `G_units`. If `G` is given, `A` must be given, and
        `N` and `K` cannot be given.
    names : list or tuple of str, default `None`, optional
        The names of the chemical species. Names are inferred if `N` or
        `A` is given as a DataFrame, in which case `names` is
        unnecessary.
    units : string or `None`, default `None`
        The units of the concentrations inputted as `c0`. The output is
        also in these units. Allowable values are {`None`, 'molar', 'M',
        'millimolar', 'mM', 'micromolar', 'uM', 'µM', 'nanomolar', 'nM',
        'picomolar', 'pM'}. If `None`, concentrations are given as mole
        fractions. The equilibrium constants given by `K` must have
        corresponding units.
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

    Returns
    -------
    c : array or DataFrame, shape c0.shape
        Equilibrium concentrations of all species. `c[i, j]` is the
        equilibrium concentration of species `j` for initial
        concentrations given by `c0[i, :]` in units given by `units`. If
        `c0` is inputted as a DataFrame or `names` is not `None`, then
        `c` is a DataFrame with columns given by `names` or with the
        same columns (without `'equilibrium constant'`) as `c0`.
        Otherwise, `c` is returned as a Numpy array with the same shape
        as `c0` with

    Other Parameters
    ----------------
    max_iters : int, default 1000
        Maximum number of iterations allowed in trust region method.
    tol : float, default 0.0000001
        Tolerance for convergence. The absolute tolerance for the
        constraints are `tol * A @ c0`.
    delta_bar : float, default 1000.0
        Maximum step size allowed in the trust region method.
    eta : float, default 0.125
        Value for eta in the trust region method. `eta` must satisfy
        `0 < eta < 0.25`.
    min_delta : float, default 1e-12
        Minimal allowed radius of the trust region. When the trust
        region radius gets below `min_delta`, the trust region
        iterations stop, and a final set of Newton steps is attempted.
    max_trials : int, default 100
        In the event that an attempt to solve does not converge, the
        solver tries again with different initial guesses.
        This continues until `max_trials` failures.
    perturb_scale : float, default 100.0
        Multiplier on random perturbations to the initial guesses
        as new ones are generated.

    Raises
    ------
    ValueError
        If input is in any way invalid
    RuntimeError
        If the trust region algorithm failed to converge

    Notes
    -----
    .. Uses an elliptical trust region optimization to find the
       equilibrium concentrations. See [1]_ for algorithmic details,
       as well as definitions of the parameters associated with the
       trust region algorithm.
    .. In practice, the trust region parameters should not be adjusted
       from their default values.

    References
    ----------
    .. [1] Nocedal and Wright, Numerical Optimization, Second Edition,
       Springer, 2006, Chapter 4.

    Examples
    --------
    1) Find the equilibrium concentrations of a solution containing
       species A, B, C, AB, BB, and BC that can undergo chemical
       reactions

                A <=> C,      K = 50 (dimensionless)
            A + C <=> AB      K = 10 (1/mM)
            B + B <=> BB      K = 40 (1/mM)
            B + C <=> BC      K = 100 (1/mM)
       with initial concentrations of [A]_0 = 1.0 mM, [B]_0 = 3.0 mM.

    >>> N = np.array([[-1,  0,  1,  0,  0,  0],
    ...               [ 1,  1,  0, -1,  0,  0],
    ...               [ 0,  2,  0,  0, -1,  0],
    ...               [ 0,  1,  1,  0,  0, -1]])
    >>> K = np.array([50.0, 0.1, 0.025, 0.01])
    >>> c0 = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    >>> eqtk.solve(c0, N=N, K=K, units='mM')
    array([0.00121271, 0.15441164, 0.06063529, 0.00187256, 0.95371818,
           0.93627945])

    2) Compute a titration curve for the same reaction system as in
       example (1) with [A]_0 = 1.0 mM. Consider B being titrated from
       [B]_0 = 0.0 to 3.0 and only use four titration points for
       display purposes.

    >>> names = ['A', 'B', 'C', 'AB', 'BB', 'BC']
    >>> df_N = pd.DataFrame(
    ...     data=[[-1,  0,  1,  0,  0,  0],
    ...           [ 1,  1,  0, -1,  0,  0],
    ...           [ 0,  2,  0,  0, -1,  0],
    ...           [ 0,  1,  1,  0,  0, -1]],
    ...     columns=names
    ... )
    >>> df_N['equilibrium constant'] = [50.0, 0.1, 0.025, 0.01]
    >>> df_c0 = pd.DataFrame(data=np.zeros((4, 6)), columns=names)
    >>> df_c0['A'] = 1.0
    >>> df_c0['B'] = np.arange(4)
    >>> df_c = eqtk.solve(df_c0, N=df_N, units='mM')
    >>> df_c.loc[:, ~df_c.columns.str.contains('__0')] # Don't disp. c0
              A         B         C        AB        BB        BC
    0  0.019608  0.000000  0.980392  0.000000  0.000000  0.000000
    1  0.003750  0.043044  0.187513  0.001614  0.074110  0.807122
    2  0.001656  0.110347  0.082804  0.001827  0.487057  0.913713
    3  0.001213  0.154412  0.060635  0.001873  0.953718  0.936279

    3) Find the equilibrium concentrations of a solution containing
       species A, B, C, AB, and AC that have free energies (in units of
       kT):
          A :   0.0
          B :   1.0
          C :  -2.0
          AB : -3.0
          AC : -7.0,
       with initial mole fractions x0_A = 0.005, x0_B = 0.001, and
       x0_C = 0.002.  The ordering of the compounds in the example is
       A, B, C, AB, AC.

    >>> A = np.array([[ 1,  0,  0,  1,  1],
    ...               [ 0,  1,  0,  1,  0],
    ...               [ 0,  0,  1,  0,  1]])
    >>> G = np.array([0.0, 1.0, -2.0, -3.0, -7.0])
    >>> c0 = np.array([0.005, 0.001, 0.002, 0.0, 0.0])
    >>> eqtk.solve(c0, A=A, G=G)
    array([0.00406569, 0.00081834, 0.00124735, 0.00018166, 0.00075265])

    4) Competition assay.  Species A binds both B and C with specified
       dissociation constants, Kd.
           AB <=> A + B,      Kd = 50 nM
           AC <=> A + C       Kd = 10 nM
       Initial concentrations of [A]_0 = 2.0 uM, [B]_0 = 0.05 uM,
       [C]_0 = 1.0 uM.  The ordering of the compounds in the example
       is A, B, C, AB, AC.

    >>> rxns = '''
    ... AB <=> A + B ; 0.05
    ... AC <=> A + C ; 0.01'''
    >>> N = eqtk.parse_rxns(rxns)
    >>> c0 = pd.Series({'A': 2.0, 'B': 0.05, 'C': 1, 'AB': 0, 'AC': 0})
    >>> eqtk.solve(c0, N=N, units='µM')
    A__0     2.000000
    B__0     0.050000
    C__0     1.000000
    AB__0    0.000000
    AC__0    0.000000
    A        0.962749
    B        0.002469
    C        0.010280
    AB       0.047531
    AC       0.989720
    dtype: float64
    """
    x0, N, K, A, G, names, solvent_density, single_point = parsers._parse_input(
        c0, N, K, A, G, names, units, solvent_density, T, G_units
    )

    # Solve for mole fractions
    if N is None:
        x = solveAG(
            A,
            G,
            x0,
            max_iters=max_iters,
            tol=tol,
            delta_bar=delta_bar,
            eta=eta,
            min_delta=min_delta,
            max_trials=max_trials,
            perturb_scale=perturb_scale,
        )
    elif G is None:
        x = solveNK(
            N,
            -np.log(K),
            x0,
            max_iters=max_iters,
            tol=tol,
            delta_bar=delta_bar,
            eta=eta,
            min_delta=min_delta,
            max_trials=max_trials,
            perturb_scale=perturb_scale,
        )
    else:
        x = solveNG(
            N,
            G,
            x0,
            max_iters=max_iters,
            tol=tol,
            delta_bar=delta_bar,
            eta=eta,
            min_delta=min_delta,
            max_trials=max_trials,
            perturb_scale=perturb_scale,
        )

    return parsers._parse_output(x, x0, names, solvent_density, single_point, units)


def volumetric_titration(
    c0,
    c0_titrant,
    vol_titrant,
    N=None,
    K=None,
    A=None,
    G=None,
    names=None,
    units=None,
    G_units=None,
    solvent_density=None,
    T=293.15,
    max_iters=1000,
    tol=0.0000001,
    delta_bar=1000.0,
    eta=0.125,
    min_delta=1.0e-12,
    max_trials=100,
    perturb_scale=100.0,
):
    """Solve for equilibrium concentrations of all species in a dilute
    solution as titrant is added.

    Parameters
    ----------
    c0 : array_like or Series, shape (n_compounds, )
        An array containing the total "initial" concentration of all
        possible compounds in solution before any titrant is added. If
        `c0` is inputted as a Pandas Series, the indices contain the
        name of the chemical species and each value is the "initial
        concentration." If `c0` is passed as a dict, the dict must be
        convertible to a Pandas Series as `pd.Series(c0)`.
    c0_titrant : array_like or Series, shape (n_compounds, )
        An array containing the total "initial" concentration of all
        possible compounds in the titrant. The titrant is itself is
        also in equilibrium, but `c0` need not contain that actual
        equilibrium concentrations in the titrant; simply the "initial"
        concentrations of all species in the titrant such that the
        equilibrium may be calculated. If `c0` is inputted as a Pandas
        Series, the indices contain the name of the chemical species and
        each value is the "initial concentration."
    vol_titrant : array_like, shape (n_titration_points, )
        Each entry is the volume of titrant added, as a fraction of the
        initial volume of the solution before addition of solution. Note
        that this is the volume of *added titrant*, not the total volume
        of the mixed solution.
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
    A : array_like or DataFrame, n_compounds columns
        Constraint matrix. If `c` is the output, then `A @ c0 = A @ c`.
        All entries must be nonnegative and the rows of `A` must be
        linearly independent. If entered as a DataFrame, the name of
        chemical species `j` is `A.columns[j]`. If `A` is given, `G`
        must be given, and `N` and `K` cannot be given.
    G : array_like, shape (n_compounds, ), default `None`
        `G[j]` is the free energy of chemical species `j` in units
        specified by `G_units`. If `G` is given, `A` must be given, and
        `N` and `K` cannot be given.
    units : string or `None`, default `None`
        The units of the concentrations inputted as `c0`. The output is
        also in these units. Allowable values are {`None`, 'molar', 'M',
        'millimolar', 'mM', 'micromolar', 'uM', 'µM', 'nanomolar', 'nM',
        'picomolar', 'pM'}. If `None`, concentrations are given as mole
        fractions. The equilibrium constants given by `K` must have
        corresponding units.
    G_units : string, default `None`
        Units in which free energy is given. If `None` or `'kT'`, the
        free  energies are specified in units of of the thermal energy
        kT. Allowable values are {None, 'kT', kcal/mol', 'J', 'J/mol',
        'kJ/mol', 'pN-nm'}.
    names : list or tuple of str, default `None`, optional
        The names of the chemical species. Names are inferred if `N` or
        `A` is given as a DataFrame, in which case `names` is
        unnecessary.
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

    Returns
    -------
    c : array or DataFrame, shape c0.shape
        Equilibrium concentrations of all species. `c[i, j]` is the
        equilibrium concentration of species `j` for initial
        concentrations given by `c0[i, :]` in units given by `units`. If
        `c0` is inputted as a DataFrame or `names` is not `None`, then
        `c` is a DataFrame with columns given by `names` or with the
        same columns (without `'equilibrium constant'`) as `c0`.
        Otherwise, `c` is returned as a Numpy array with the same shape
        as `c0` with

    Other Parameters
    ----------------
    max_iters : int, default 1000
        Maximum number of iterations allowed in trust region method.
    tol : float, default 0.0000001
        Tolerance for convergence. The absolute tolerance for the
        constraints are `tol * A @ c0`.
    delta_bar : float, default 1000.0
        Maximum step size allowed in the trust region method.
    eta : float, default 0.125
        Value for eta in the trust region method. `eta` must satisfy
        `0 < eta < 0.25`.
    min_delta : float, default 1e-12
        Minimal allowed radius of the trust region. When the trust
        region radius gets below `min_delta`, the trust region
        iterations stop, and a final set of Newton steps is attempted.
    max_trials : int, default 100
        In the event that an attempt to solve does not converge, the
        solver tries again with different initial guesses.
        This continues until `max_trials` failures.
    perturb_scale : float, default 100.0
        Multiplier on random perturbations to the initial guesses
        as new ones are generated.

    Raises
    ------
    ValueError
        If input is in any way invalid
    RuntimeError
        If the trust region algorithm failed to converge

    Notes
    -----
    .. Uses an elliptical trust region optimization to find the
       equilibrium concentrations. See [1]_ for algorithmic details,
       as well as definitions of the parameters associated with the
       trust region algorithm.
    .. In practice, the trust region parameters should not be adjusted
       from their default values.

    References
    ----------
    .. [1] Nocedal and Wright, Numerical Optimization, Second Edition,
       Springer, 2006, Chapter 4.

    Examples
    --------
    Compute the pH titration curve of a 1 M solution of weak acid HA
    with acid dissociation constant 1e-5 M, titrating in 1.0 M NaOH.

    >>> rxns = '''
    ... <=> H+ + OH- ; 1e-14
    ... HA <=> H+ + A- ; 1e-5'''
    >>> N = eqtk.parse_rxns(rxns)
    >>> c0 = pd.Series([0, 0, 1, 0], index=['H+', 'OH-', 'HA', 'A-'])
    >>> c0_titrant = pd.Series([0, 1, 0, 0],
                               index=['H+', 'OH-', 'HA', 'A-'])
    >>> vol_titrant = np.array([0, 1, 2]) # Only few for display
    >>> c = eqtk.volumetric_titration(c0, c0_titrant, vol_titrant, N=N,
    ...     units="M")
    >>> c['pH'] = -np.log10(c['H+'])
    >>> c[['vol titrant / initial vol', 'pH']]
       vol titrant / initial vol         pH
    0                          0   2.500687
    1                          1   9.349480
    2                          2  13.522879


    """
    if np.any(vol_titrant < 0):
        raise ValueError("`vol_titrant` must have non-negative volumes.")

    x0, N, K, A, G, names, solvent_density, single_point = parsers._parse_input(
        c0, N, K, A, G, names, units, solvent_density, T, G_units
    )

    if x0.shape[0] != 1:
        raise ValueError("`c0` must be a one-dimensional array.")

    x0_titrant, _, _, _, _, _, _, _ = parsers._parse_input(
        c0_titrant, N, K, A, G, names, units, solvent_density, T, G_units
    )

    if x0_titrant.shape[0] != 1:
        raise ValueError("`c0_titrant` must be a one-dimensional array.")

    new_x0 = _volumetric_to_c0(x0.flatten(), x0_titrant.flatten(), vol_titrant)

    if N is None:
        x = solveAG(
            A,
            G,
            x0,
            max_iters=max_iters,
            tol=tol,
            delta_bar=delta_bar,
            eta=eta,
            min_delta=min_delta,
            max_trials=max_trials,
            perturb_scale=perturb_scale,
        )
    elif G is None:
        x = solveNK(
            N,
            -np.log(K),
            x0,
            max_iters=max_iters,
            tol=tol,
            delta_bar=delta_bar,
            eta=eta,
            min_delta=min_delta,
            max_trials=max_trials,
            perturb_scale=perturb_scale,
        )
    else:
        x = solveNG(
            N,
            G,
            x0,
            max_iters=max_iters,
            tol=tol,
            delta_bar=delta_bar,
            eta=eta,
            min_delta=min_delta,
            max_trials=max_trials,
            perturb_scale=perturb_scale,
        )

    c = parsers._parse_output(
        x, new_x0 * solvent_density, names, solvent_density, False, units
    )

    if type(c) == pd.core.frame.DataFrame:
        c["vol titrant / initial vol"] = vol_titrant

    return c


def fixed_value_solve(
    c0,
    fixed_c,
    N=None,
    K=None,
    A=None,
    G=None,
    names=None,
    units=None,
    G_units=None,
    solvent_density=None,
    T=293.15,
    max_iters=1000,
    tol=0.0000001,
    delta_bar=1000.0,
    eta=0.125,
    min_delta=1.0e-12,
    max_trials=100,
    perturb_scale=100.0,
):
    """
    """
    x0, N, K, A, G, names, solvent_density, _ = parsers._parse_input(
        c0, N, K, A, G, names, units, solvent_density, T, G_units
    )

    c0_from_df = type(c0) in [pd.core.frame.DataFrame, pd.core.series.Series]

    fixed_x, x0, single_point = parsers._parse_fixed_c(
        fixed_c, x0, c0_from_df, names, solvent_density
    )

    # Convert the problem to N, K
    if G is not None:
        # Strategy:
        # Prune the problem for the A, G formulation, and then convert it to N, K using
        #        N = linalg.nullspace_svd(A, tol=constants.nullspace_tol)
        #        K = np.exp(-np.dot(N, G))
        # and then do NOT prune the N, K problem, and solve.

        raise NotImplementedError(
            "Fixed value solving not yet implemented for the A, G formulation."
        )

    x = np.empty_like(x0)
    for i, (fixed_x_row, x0_row) in enumerate(zip(fixed_x, x0)):
        N_new, K_new = _new_NK_fixed_x(fixed_x_row, N, K)
        print(N_new, K_new)
        x_res = solveNK(
            N_new,
            -np.log(K_new),
            np.ascontiguousarray(np.expand_dims(x0_row, axis=0)),
            max_iters=max_iters,
            tol=tol,
            delta_bar=delta_bar,
            eta=eta,
            min_delta=min_delta,
            max_trials=max_trials,
            perturb_scale=perturb_scale,
        )
        x[i] = x_res[0]

    c = parsers._parse_output(
        x, x0 * solvent_density, names, solvent_density, single_point, units
    )

    if type(c) == pd.core.series.Series:
        for j in np.nonzero(~np.isnan(fixed_x))[0]:
            c[f"[{names[j]}]__0 ({units})"] = np.nan
            c[f"[{names[j]}]__fixed ({units})"] = fixed_x[0, j] * solvent_density

    if type(c) == pd.core.frame.DataFrame:
        cols = [f"[{names[j]}]__fixed ({units})" for name in names]
        data = np.empty((len(c), len(names)))
        data = np.fill(np.nan)
        c = pd.concat(
            (c, pd.DataFrame(data=data, columns=cols)), axis=1, ignore_index=True
        )
        for i in range(len(c)):
            for j in np.nonzero(~np.isnan(fixed_x))[0]:
                c.loc[f"[{names[j]}]__0 ({units})"] = np.nan
                c.loc[i, f"[{names[j]}]__fixed ({units})"] = (
                    fixed_x[i, j] * solvent_density
                )
        c = c.dropna(axis=1, how="all")

    return c


def _new_NK_fixed_x(fixed_x, N, K):
    """Generate a new N and K for a fixed x.

    fixed_x is a 1D array.
    """
    N_new = N.copy()
    K_new = K.copy()

    for j in np.nonzero(~np.isnan(fixed_x))[0]:
        N_new_row = np.zeros(N.shape[1])
        N_new_row[j] = 1.0
        N_new = np.vstack((N_new_row, N_new))
        K_new = np.concatenate(((fixed_x[j],), K_new))

    if N_new.shape[0] > N_new.shape[1]:
        raise ValueError(
            "Cannot fix concentration as specified: Results in an over-constrained problem."
        )

    if np.linalg.matrix_rank(N_new) != N_new.shape[0]:
        raise ValueError(
            "Cannot fix concentration as specified: Results in a rank-deficient stoichiometic matrix."
        )

    return np.ascontiguousarray(N_new), np.ascontiguousarray(K_new)


def to_df(c, c0=None, units=None, names=None):
    """
    Return output as a Pandas DataFrame.
    """
    units_str = " (" + units + ")" if units is not None else ""

    if c0 is None:
        if names is None:
            names = ["species_" + str(i) for i in range(c.shape[1])]
        cols = [name + units_str for name in names]

        return pd.DataFrame(data=c, columns=cols)

    if type(c) == pd.core.series.Series:
        if len(c.index) == 2 * len(c0.index) and np.sum(
            c.index.str.contains("__0")
        ) == len(c0.index):
            return c.copy().rename(index=lambda x: x + units_str)
    elif type(c) == pd.core.frame.DataFrame:
        if len(c.columns) == 2 * len(c0.columns) and np.sum(
            c.columns.str.contains("__0")
        ) == len(c0.columns):
            return c.copy().rename(columns=[col + units_str for col in c.columns])

    parsers._check_names_type(names)

    c0, n_compounds, names, _, single_point = parsers._parse_c0(c0, names)

    if names is None:
        names = ["species_" + str(i) for i in range(n_compounds)]

    c, _, _, _, _ = parsers._parse_c0(c, names)

    if c0.shape != c.shape:
        raise ValueError("`c0` and `c` have mismatched shapes.")

    cols = [name + "__0" + units_str for name in names]
    cols += [name + units_str for name in names]

    if single_point:
        return pd.Series(index=cols, data=np.concatenate((c0.flatten(), c.flatten())))

    return pd.DataFrame(columns=cols, data=np.concatenate((c0, c), axis=1))


def _volumetric_to_c0(c0, c0_titrant, vol_titrant):
    """Convert volumetric titration to input concentrations.

    Parameters
    ----------
    c0 : array, shape (n_compounds, )
        An array containing the total "initial" concentration of all
        possible compounds in solution before any titrant is added.
    c0_titrant : array, shape (n_compounds, )
        An array containing the total "initial" concentration of all
        possible compounds in the titrant.
    vol_titrant : array_like, shape (n_titration_points, )
        Each entry is the volume of titrant added, as a fraction of the
        initial volume of the solution before addition of solution.

    Returns
    -------
    output : array, shape(n_titration_points, n_compounds)
        Values of "initial concentrations" for all species in the dilute
        solution after addition of the titrant. The output of this
        function is used as `c0` in `solve()` to get the results for a
        volumetric titration.
    """
    n_titration_points = len(vol_titrant)
    n_compounds = len(c0_titrant)
    if n_compounds != len(c0):
        raise ValueError("Dimensions of `c0` and `c0_titrant` must agree.")

    vol_titrant_col = np.reshape(vol_titrant, (n_titration_points, 1))
    c_titrant_row = np.reshape(c0_titrant, (1, n_compounds))
    c0_row = np.reshape(c0, (1, n_compounds))
    titrated_scale = vol_titrant_col / (1 + vol_titrant_col)
    original_scale = 1 / (1 + vol_titrant_col)

    return np.dot(titrated_scale, c_titrant_row) + np.dot(original_scale, c0_row)


@jit("double[::1](double[::1], boolean[::1], int64)", nopython=True)
def _boolean_index(a, b, n_true):
    """Returns a[b] where b is a Boolean array."""
    # if n_true == 0:
    #     return np.array([np.float64(x) for x in range(0)])

    out = np.empty(n_true)
    j = 0
    for i, tf in enumerate(b):
        if tf:
            out[j] = a[i]
            j += 1

    return out


@jit(
    "double[:, ::1](double[:, ::1], boolean[::1], boolean[::1], int64, int64)",
    nopython=True,
)
def _boolean_index_2d(a, b_row, b_col, n_true_row, n_true_col):
    """Does the following:
    a_new = a[b_row, :]
    a_new = a_new[:, b_col]
    """
    out = np.empty((n_true_row, n_true_col))
    m = 0
    for i, tf_row in enumerate(b_row):
        if tf_row:
            n = 0
            for j, tf_col in enumerate(b_col):
                if tf_col:
                    out[m, n] = a[i, j]
                    n += 1
            m += 1

    return out


@jit("double[::1](double[::1], double[::1], double[:, ::1])", nopython=True)
def _initial_guess(constraint_vector, G, A):
    """
    Calculates an initial guess for lambda such that the maximum
    mole fraction calculated will not give an overflow error and
    the objective function $-h(µ)$ will be positive.  It is
    best to have a positive objective function because when the
    objective function is negative, it tends to be very close to
    zero and there are precision issues.

    We assume all the mu's have the same value in the initial
    condition.  We compute the maximal lambda such that all mole
    fractions of all complexes are below some maximum.
    """
    # OLD WAY
    # mu_guess = ((1.0 + G) / abs(A).sum(0)).min() \
    #    * ones_like(constraint_vector)

    # Guess mu such that ln x = 1 for all x (x ~ 3).
    A_AT = np.dot(A, A.transpose())
    b = np.dot(A, G + 1.0)
    mu0, success = linalg.solve_pos_def(A_AT, b)
    if success:
        return mu0
    else:
        return np.linalg.solve(A_AT, b)


@jit(
    "double[::1](double[::1], double[::1], double[:, ::1], double[::1], double)",
    nopython=True,
)
def _perturb_initial_guess(constraint_vector, G, A, mu0, perturb_scale=100.0):
    """
    Calculates an initial guess for lambda such that the maximum
    mole fraction calculated will not give an overflow error and
    the objective function $-h(µ)$ will be positive.  It is
    best to have a positive objective function because when the
    objective function is negative, it tends to be very close to
    zero and there are precision issues.

    We assume all the mu's have the same value in the initial
    condition.  We compute the maximal lambda such that all mole
    fractions of all complexes are below some maximum.
    """
    # OLD WAY
    # mu_guess = ((1.0 + G) / abs(A).sum(0)).min() \
    #    * ones_like(constraint_vector)

    new_mu = mu0 + perturb_scale * 2.0 * (np.random.rand(len(constraint_vector)) - 0.5)
    # Prevent overflow err
    while (-G + np.dot(new_mu, A)).max() > constants.max_logx:
        perturb_scale /= 2.0

    return new_mu


@jit(
    "Tuple((double[::1], boolean, int64, int64[::1]))(double[:, ::1], double[::1], double[::1], int64, double, double, double, double, int64, double)",
    nopython=True,
)
def _solve_trust_region(
    A,
    G,
    constraint_vector,
    max_iters=1000,
    tol=0.0000001,
    delta_bar=1000.0,
    eta=0.125,
    min_delta=1.0e-12,
    max_trials=100,
    perturb_scale=100.0,
):
    """
    Solve for equilibrium concentrations of all species in a dilute
    solution given their free energies in units of kT.

    Parameters
    ----------
    A : array_like, shape (n_constraints, n_compounds)
        Each row represents a system constraint.
        Namely, np.dot(A, x) = constraint_vector.
    G : array_like, shape (n_compounds,)
        G[j] is the free energy of compound j in units of kT.
    constraint_vector : array_like, shape (n_constraints,)
        Right hand since of constraint equation,
        np.dot(A, x) = constraint_vector.

    Returns
    -------
    x : array_like, shape (n_compounds)
        x[j] = the equilibrium mole fraction of compound j.
    converged : Boolean
        True is trust region calculation converged, False otherwise.
    run_stats : a class with attributes:
        Statistics of steps taken by the trust region algorithm.
        n_newton_steps : # of Newton steps taken
        n_cauchy_steps : # of Cauchy steps taken (hit trust region boundary)
        n_dogleg_steps : # of dogleg steps taken
        n_chol_fail_cauchy_steps : # of Cholesky failures forcing Cauchy step
        n_irrel_chol_fail : # of steps with irrelevant Cholesky failures
        n_dogleg_fail : # of failed dogleg calculations

    Raises
    ------
    ValueError
        If A, G, and constraint_vector have a dimensional mismatch.
    """
    # Dimension of the problem
    n_constraints, n_compounds = A.shape

    # Build new problem with inactive ones cut out
    params = (G, A, constraint_vector)
    abs_tol = tol * np.abs(constraint_vector)
    mu0 = _initial_guess(constraint_vector, G, A)

    mu, converged, step_tally = trust_region.trust_region_convex_unconstrained(
        mu0,
        G,
        A,
        constraint_vector,
        tol=abs_tol,
        max_iters=max_iters,
        delta_bar=delta_bar,
        eta=eta,
        min_delta=min_delta,
    )

    # Try other initial guesses if it did not converge
    n_trial = 1
    while not converged and n_trial < max_trials:
        mu0 = _perturb_initial_guess(constraint_vector, G, A, mu0, perturb_scale)
        mu, converged, step_tally = trust_region.trust_region_convex_unconstrained(
            mu0,
            G,
            A,
            constraint_vector,
            tol=abs_tol,
            max_iters=max_iters,
            delta_bar=delta_bar,
            eta=eta,
            min_delta=min_delta,
        )
        n_trial += 1

    x = np.exp(trust_region.compute_logx(mu, G, A))

    return x, converged, n_trial, step_tally


@jit(
    "Tuple((double[:, ::1], double[::1], double[::1], boolean[::1], boolean[::1]))(double[:, ::1], double[::1], double[::1])",
    nopython=True,
)
def _prune_NK(N, minus_log_K, x0):
    """Prune reactions to ignore inert and missing species.
    """
    n_reactions, n_compounds = N.shape

    prev_active = x0 > 0
    active_compounds = x0 > 0
    active_reactions = np.zeros(n_reactions, dtype=np.bool8)
    done = False
    n_reactions = N.shape[0]

    while not done:
        for i in range(n_reactions):
            f_rate = 1
            b_rate = 1
            for j in range(n_compounds):
                if N[i, j] < 0:
                    f_rate *= active_compounds[j]
                if N[i, j] > 0:
                    b_rate *= active_compounds[j]
            if f_rate > 0:
                active_reactions[i] = True
                for j in range(n_compounds):
                    if N[i, j] > 0:
                        active_compounds[j] = True
            if b_rate > 0:
                active_reactions[i] = True
                for j in range(n_compounds):
                    if N[i, j] < 0:
                        active_compounds[j] = True
        done = np.all(active_compounds == prev_active)
        prev_active = np.copy(active_compounds)

    # Select all compounds that are in at least one active reaction
    # Can be calc'ed as np.dot(active_reactions, N != 0), but that's not numba-able
    active_compounds = np.empty(n_compounds, dtype=np.bool8)
    nonzero_N = N != 0
    nonzero_N_T = nonzero_N.transpose()
    for i in range(nonzero_N_T.shape[0]):
        active_compounds[i] = np.any(
            np.logical_and(active_reactions, nonzero_N_T[i, :])
        )

    # active reactions and compounds are now bools, so sum to get numbers
    n_reactions_new = np.sum(active_reactions)
    n_compounds_new = np.sum(active_compounds)

    if n_reactions_new > 0 and n_compounds_new > 0:
        minus_log_K_new = _boolean_index(minus_log_K, active_reactions, n_reactions_new)
        N_new = _boolean_index_2d(
            N, active_reactions, active_compounds, n_reactions_new, n_compounds_new
        )
        x0_new = _boolean_index(x0, active_compounds, n_compounds_new)
    else:
        # Use trick to get typed empty list
        # http://numba.pydata.org/numba-doc/latest/user/troubleshoot.html#my-code-has-an-untyped-list-problem
        N_new = np.array([np.float64(x) for x in range(0)]).reshape((1, 0))
        minus_log_K_new = np.array([np.float64(x) for x in range(0)])
        x0_new = x0

    return N_new, minus_log_K_new, x0_new, active_compounds, active_reactions


@jit(
    "Tuple((double[:, ::1], double[::1], double[::1], boolean[::1]))(double[:, ::1], double[::1], double[::1])",
    nopython=True,
)
def _prune_AG(A, G, x0):
    """Prune constraint matrix and free energy to ignore inert and
    missing species.
    """
    constraint_vector = np.dot(A, x0)

    active_constraints = constraint_vector > 0.0
    active_compounds = np.ones(len(x0), dtype=np.bool8)

    for i, act_const in enumerate(active_constraints):
        if not act_const:
            for j in range(A.shape[1]):
                if A[i, j] > 0.0:
                    active_compounds[j] = False

    n_active_constraints = np.sum(active_constraints)
    n_active_compounds = np.sum(active_compounds)

    A_new = _boolean_index_2d(
        A,
        active_constraints,
        active_compounds,
        n_active_constraints,
        n_active_compounds,
    )

    constraint_vector_new = _boolean_index(
        constraint_vector, active_constraints, n_active_constraints
    )

    G_new = _boolean_index(G, active_compounds, np.sum(active_compounds))

    return A_new, G_new, constraint_vector_new, active_compounds


@jit(
    "void(double[:, ::1], double[::1], double[:, ::1], double[::1], int64, int64, double, double, double, double, double, boolean, int64[::1], int64)",
    nopython=True,
)
def _print_runstats(
    A,
    G,
    x0,
    constraint_vector,
    n_trial,
    max_trials,
    tol,
    delta_bar,
    eta,
    min_delta,
    perturb_scale,
    converged,
    step_tally,
    max_iters,
):
    print("RUN STATS:")
    print("  constraint matrix A:", A)
    print("  compound free energies G:", G)
    print("  x0:", x0)
    print("  number of constraints:", len(constraint_vector))
    print("  constraint vector np.dot(A, x0):", constraint_vector)
    print("  number of attempts:", n_trial)
    print("  maximum allowed number of attempts:", max_trials)
    print("  tolerance:", tol)
    print("  delta_bar:", delta_bar)
    print("  eta:", eta)
    print("  minimum allowed delta:", min_delta)
    print("  scale for perturbing initial guesses:", perturb_scale)
    print("  RESULTS FROM LAST ATTEMPT:")
    print("    converged:", converged)
    print("    number of iterations:", np.sum(step_tally))
    print("    number of Newton steps:", step_tally[0])
    print("    number of Cauchy steps:", step_tally[1])
    print("    number of dogleg steps:", step_tally[2])
    print("    number of Cholesky failures forcing a Cauchy step:", step_tally[3])
    print("    number of irrelevant Cholesky failures:", step_tally[4])
    print("    number of dogleg failures:", step_tally[5])


@jit("double[::1](double[:, ::1], double[::1])", nopython=True)
def _create_from_nothing(N, x0):
    for i in range(N.shape[0]):
        Ni = N[i, :]
        if np.all(Ni >= 0):
            x0 += Ni
        elif np.all(Ni <= 0):
            x0 -= Ni

    return x0


@jit(nopython=True)
def solveNK(
    N,
    minus_log_K,
    x0,
    max_iters=1000,
    tol=0.0000001,
    delta_bar=1000.0,
    eta=0.125,
    min_delta=1.0e-12,
    max_trials=100,
    perturb_scale=100.0,
):
    """
    Solve for equilibrium concentrations of all species in a dilute
    solution.

    Parameters
    ----------
    N : array_like, shape (n_reactions, n_compounds)
        N[r][j] = the stoichiometric coefficient of compounds j
        in chemical reaction r.
    minus_log_K : array_like, shape (n_reactions,)
        minus_log_K[r] is the minus log of the equilibrium constant for
        chemical reaction r
    x0 : array_like, shape (n_points, n_compounds)
        array containing the total "initial" mole fraction of all
        compounds in solution.  Internally, this is converted into
        n_compounds-n_reactions conservation equations.

    Returns
    -------
    x : array_like, shape (n_compounds)
        x[j] = the equilibrium concentration of compound j.  Units are
        given as specified in the units keyword argument.

    Raises
    ------
    ValueError
        If input is in any way invalid.
    RuntimeError
        If the trust region algorithm failed to converge.

    Notes
    -----
    .. N must have full row rank, i.e., all rows must be
       linearly independent.

    .. All x0's must be non-negative and finite

    """
    # Get number of particles and compounds
    n_reactions, n_compounds = N.shape

    x = np.empty_like(x0)

    for i_point in range(x0.shape[0]):
        N_new, minus_log_K_new, x0_new, active_compounds, _ = _prune_NK(
            N, minus_log_K, x0[i_point]
        )

        if len(minus_log_K_new) > 0:
            n_reactions_new, n_compounds_new = N_new.shape
            n_constraints_new = n_compounds_new - n_reactions_new

            # Compute and check constraint matrix
            A = linalg.nullspace_svd(N_new, tol=constants.nullspace_tol)

            # If completely constrained (N square), solve directly
            if n_constraints_new == 0:
                ln_x = np.linalg.solve(N_new, -minus_log_K_new)
                x_new = np.exp(ln_x)
            else:
                # In case we have null <=> compds type reaction, adjust x0
                x0_adjusted = _create_from_nothing(N_new, x0_new)

                # Compute the free energies in units of kT from the K's
                b = np.concatenate((np.zeros(n_constraints_new), minus_log_K_new))
                N_prime = np.vstack((A, N_new))
                G = np.linalg.solve(N_prime, b)
                constraint_vector = np.dot(A, x0_adjusted)

                x_new, converged, n_trial, step_tally = _solve_trust_region(
                    A,
                    G,
                    constraint_vector,
                    max_iters=max_iters,
                    tol=tol,
                    delta_bar=1000.0,
                    eta=eta,
                    min_delta=min_delta,
                    max_trials=max_trials,
                    perturb_scale=perturb_scale,
                )

                # If not converged, throw exception
                if not converged:
                    print("**** Convergence failure! ****")
                    _print_runstats(
                        A,
                        G,
                        x0,
                        constraint_vector,
                        n_trial,
                        max_trials,
                        tol,
                        delta_bar,
                        eta,
                        min_delta,
                        perturb_scale,
                        converged,
                        step_tally,
                        max_iters,
                    )
                    raise RuntimeError("Calculation did not converge")

        # Put in concentrations that were cut out
        j = 0
        for i in range(n_compounds):
            if active_compounds[i]:
                x[i_point, i] = x_new[j]
                j += 1
            else:
                x[i_point, i] = x0[i_point, i]

    return x


@jit(nopython=True)
def solveNG(
    N,
    G,
    x0,
    max_iters=1000,
    tol=0.0000001,
    delta_bar=1000.0,
    eta=0.125,
    min_delta=1.0e-12,
    max_trials=100,
    perturb_scale=100.0,
):
    """
    Solve for equilibrium concentrations of all species in a dilute
    solution.

    Parameters
    ----------
    N : array_like, shape (n_reactions, n_compounds)
        N[r][j] = the stoichiometric coefficient of compounds j
        in chemical reaction r.
    G : array_like, shape (n_compounds,)
        G[j] is the free energy of compound j in units of kT.
    x0 : array_like, shape (n_points, n_compounds)
        array containing the total "initial" mole fraction of all
        compounds in solution.  Internally, this is converted into
        n_compounds-n_reactions conservation equations.

    Returns
    -------
    x : array_like, shape (n_compounds)
        x[j] = the equilibrium concentration of compound j.  Units are
        given as specified in the units keyword argument.

    Raises
    ------
    ValueError
        If input is in any way invalid.
    RuntimeError
        If the trust region algorithm failed to converge.

    Notes
    -----
    .. N must have full row rank, i.e., all rows must be
       linearly independent.

    .. All x0's must be non-negative and finite

    """
    # Get number of particles and compounds
    n_reactions, n_compounds = N.shape

    x = np.empty_like(x0)

    dummy_minus_log_K = np.ones(N.shape[0], dtype=float)

    for i_point in range(x0.shape[0]):
        N_new, dummy_throwaway, x0_new, active_compounds, _ = _prune_NK(
            N, dummy_minus_log_K, x0[i_point]
        )
        G_new = _boolean_index(G, active_compounds, np.sum(active_compounds))

        if N_new.shape[0] > 0:
            n_reactions_new, n_compounds_new = N_new.shape
            n_constraints_new = n_compounds_new - n_reactions_new

            # Compute and check constraint matrix
            A = linalg.nullspace_svd(N_new, tol=constants.nullspace_tol)

            # If completely constrained (N square), solve directly
            if n_constraints_new == 0:
                x_new = np.exp(-G_new)
            else:
                # In case we have null <=> compds type reaction, adjust x0
                x0_adjusted = _create_from_nothing(N_new, x0_new)

                constraint_vector = np.dot(A, x0_adjusted)

                x_new, converged, n_trial, step_tally = _solve_trust_region(
                    A,
                    G_new,
                    constraint_vector,
                    max_iters=max_iters,
                    tol=tol,
                    delta_bar=1000.0,
                    eta=eta,
                    min_delta=min_delta,
                    max_trials=max_trials,
                    perturb_scale=perturb_scale,
                )

                # If not converged, throw exception
                if not converged:
                    print("**** Convergence failure! ****")
                    _print_runstats(
                        A,
                        G,
                        x0,
                        constraint_vector,
                        n_trial,
                        max_trials,
                        tol,
                        delta_bar,
                        eta,
                        min_delta,
                        perturb_scale,
                        converged,
                        step_tally,
                        max_iters,
                    )
                    raise RuntimeError("Calculation did not converge")

        # Put in concentrations that were cut out
        j = 0
        for i in range(n_compounds):
            if active_compounds[i]:
                x[i_point, i] = x_new[j]
                j += 1
            else:
                x[i_point, i] = x0[i_point, i]

    return x


@jit(nopython=True)
def solveAG(
    A,
    G,
    x0,
    max_iters=1000,
    tol=0.0000001,
    delta_bar=1000.0,
    eta=0.125,
    min_delta=1.0e-12,
    max_trials=100,
    perturb_scale=100.0,
):
    """
    Solve for equilibrium concentrations of all species in a dilute
    solution given their free energies in units of kT.

    Parameters
    ----------
    A : array_like, shape (n_particles, n_compounds)
        A[i][j] = number of particles of type i in compound j.
        Leftmost square matrix of A (n_particles by n_particles) must
        be the identity matrix.  No column may be repeated.
    G : array_like, shape (n_compounds,)
        G[j] is the free energy of compound j in units of kT.
    x0 : array_like, shape (n_points, n_compounds)
        Array containing the total "initial" mole fraction of all
        compounds in solution.
    elemental : bool
        If True, A is assumed to be an elemental conservation matrix.
        This means entry A[i, j] is the number of particles of type i
        in compound j.

    Returns
    -------
    x : array_like, shape (n_compounds)
        x[j] = the equilibrium mole fraction of compound j.
    run_stats : a class with attributes:
        Statistics of steps taken by the trust region algorithm.
        n_newton_steps : # of Newton steps taken
        n_cauchy_steps : # of Cauchy steps taken (hit trust region boundary)
        n_dogleg_steps : # of dogleg steps taken
        n_chol_fail_cauchy_steps : # of Cholesky failures forcing Cauchy step
        n_irrel_chol_fail : # of steps with irrelovant Cholesky failures
        n_dogleg_fail : # of failed dogleg calculations

    Raises
    ------
    ValueError
        If A, G, and x0 have a dimensional mismatch.
    RuntimeError
        If trust region algorithm failed to converge.
    """
    n_particles, n_compounds = A.shape

    x = np.empty_like(x0)

    for i_point in range(x0.shape[0]):
        A_new, G_new, constraint_vector, active_compounds = _prune_AG(A, G, x0[i_point])

        # Detect if A is empty (no constraints)
        A_empty = A_new.shape[0] + A_new.shape[1] == 1

        # Problem is entirely constrained, must have x = x0.
        if (not A_empty) and A_new.shape[0] >= A.shape[1]:
            x[i_point, :] = x0
            converged = True
            run_stats = None
        else:
            # No constraints, directly use analytical solution to primal problem
            if A_empty:
                x_new = np.exp(-G_new)
                converged = True
                run_stats = None
            # Go ahead and solve
            else:
                x_new, converged, n_trial, step_tally = _solve_trust_region(
                    A_new,
                    G_new,
                    constraint_vector,
                    max_iters=max_iters,
                    tol=tol,
                    delta_bar=1000.0,
                    eta=eta,
                    min_delta=min_delta,
                    max_trials=max_trials,
                    perturb_scale=perturb_scale,
                )

                # If not converged, throw exception
                if not converged:
                    print("**** Convergence failure! ****")
                    _print_runstats(
                        A,
                        G,
                        x0,
                        constraint_vector,
                        n_trial,
                        max_trials,
                        tol,
                        delta_bar,
                        eta,
                        min_delta,
                        perturb_scale,
                        converged,
                        step_tally,
                        max_iters,
                    )
                    raise RuntimeError("Calculation did not converge")

            # Put in concentrations that were cut out
            j = 0
            for i in range(n_compounds):
                if active_compounds[i]:
                    x[i_point, i] = x_new[j]
                    j += 1
                else:
                    x[i_point, i] = x0[i_point, i]

    return x
