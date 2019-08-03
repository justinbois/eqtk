"""
Utilities to compute coupled equilibria in dilute solutions.
"""

import time
import warnings

import numpy as np
from . import checks
from . import trust_region
from . import linalg
from . import numba_check
from . import constants


def solve(
    c0,
    N=None,
    K=None,
    A=None,
    G=None,
    units=None,
    solvent_density=None,
    T=20.0,
    G_units=None,
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
    c0 : array_like, shape (n_points, n_compounds) or (n_compounds, )
        Each row contains the total "initial" concentration of all
        possible compounds in solution. The equilibrium concentration
        of all species is computed for each row in c0. c0[i, j] is the
        initial concentration of compound j for calculation i.
    N : array_like, shape (n_reactions, n_compounds)
        `N[r, j]` = the stoichiometric coefficient of compound j
        in chemical reaction r.  Ignored if `G` is not None.
    K : array_like, shape (n_reactions,)
        `K[r]` is the equilibrium constant for chemical reaction r.
        Ignored if G is not None.
    A : array_like, shape (n_constraints, n_compounds)
        Constraint matrix. All rows must be linearly independent.
        Ignored if G is None.
    G : array_like, shape (n_compounds, ), default None
        G[j] is the free energy, either in units kT or in units specified
        by G_units.  If not None, A must also be not None.  If this is
        the case, overrides any input for N and K.
    units : string or None, default None
        The units of the given concentrations. Allowable values are
        'molar', 'M', 'millimolar', 'mM', 'micromolar', 'uM',
        'nanomolar', 'nM', 'picomolar', 'pM', None. If None, concen-
        trations are given as mole fractions. The equilbrium constants
        given by K have corresponding units. The output is also given
        in these units.
    solvent_density : float, default = None
        The density of the solvent in units commensurate with the units
        keyword argument.  Default (None) assumes the solvent is water,
        and its density is computed at the temperature specified by the
        T keyword argument.
    T : float, default = 20.0
        Temperature, in deg. C, of the solution.  Not relevant when
        units and G_units is None,
    G_units : string, default None
        Units in which free energy is given.  If None, the free energies are
        specified in units of kT.  Other acceptable options are: 'kcal/mol',
        'J', 'J/mol', 'kJ/mol', and 'pN-nm'.
    return_run_stats : Boolean, default = False
        If True, also returns a list of statistics on steps taken by
        the trust region algorithm.
    max_iters : int, default = 1000
        Maximum number of interations allowed in trust region
        method.
    tol : float, default = 0.0000001
        Tolerance for convergence.  The absolute tolerance is
        tol * (mininium single-species initial mole fraction)
    delta_bar : float, default = 1000.0
        Maximum step size allowed in the trust region method.
    eta : float, default = 0.125
        Value for eta in the trust region method (see Nocedal and
        Wright reference). 0 < eta < 1/4.
    min_delta : float, default = 1e-12
        Minimal allowed radius of the trust region.  When the trust region
        radius gets below min_delta, the trust region iterations stop,
        and a final Newton step is attempted.
    max_trials : int, default 100
        In the event that an attempt to solve does not converge, we try
        again with different initial guesses. This continues until
        `max_trials` failures.
    perturb_scale : float, default 100.0
        Multiplier on random perturbations to the initial guesses
        as new ones are generated.

    Returns
    -------
    c : array_like, shape (n_titration_points, n_compounds)
        c[i,j] = the equilibrium concentration of compound j at titration
        point i.  Units are given as specified in the units keyword argument.
    run_stats (optional, if return_run_stats is True) : a class containing:
        statistics of steps taken by the trust region algorithm.
          n_newton_steps : # of Newton steps taken
          n_cauchy_steps : # of Cauchy steps taken (hit trust region boundary)
          n_dogleg_steps : # of dogleg steps taken
          n_chol_fail_cauchy_steps : # of Cholesky failures forcing Cauchy step
          n_irrel_chol_fail : # of steps with irrelovant Cholesky failures
          n_dogleg_fail : # of failed dogleg calculations

    Raises
    ------
    ValueError
        If input is in any way invalid
    RuntimeError
        If the trust region algorithm failed to converge

    Notes
    -----
    .. If the C library functions are unavailable, uses
       a pure Python/NumPy trust region solver.

    .. N must have full row rank, i.e., all rows must be
       linearly independent.

    .. All K's must be positive and finite.

    .. All c0's must be nonnegative and finite.

    References
    ----------
    .. [1] J.S. Bois, Analysis of nucleic acids in dilute solutions,
           Caltech Ph.D. thesis, 2007.

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

    >>> import numpy as np
    >>> import eqtk
    >>> N = np.array([[-1,  0,  1,  0,  0,  0],
                      [-1, -1,  0,  1,  0,  0],
                      [ 0, -2,  0,  0,  1,  0],
                      [ 0, -1, -1,  0,  0,  1]])
    >>> K = np.array([50.0, 10.0, 40.0, 100.0])
    >>> c0 = np.array([[1.0, 3.0, 0.0, 0.0, 0.0, 0.0]])
    >>> c = eqtk.sweep_titration(N, K, c0, units='mM')
    >>> c
    array([ 0.00121271,  0.15441164,  0.06063529,  0.00187256,  0.95371818,
            0.93627945])

    2) Find the titration curve for a solution containing
       species A, B, C, AB, BB, and BC that can undergo chemical
       reactions

                A <--> C,      K = 50 (dimensionless)
            A + C <--> AB      K = 10 (1/mM)
            B + B <--> BB      K = 40 (1/mM)
            B + C <--> BC      K = 100 (1/mM)

       with x_A0 = 0.0001.  We consider B being titrated from
       x_B0 = 0.0 to x_B0 = 0.001.

    >>> import numpy as np
    >>> import eqtk
    >>> N = np.array([[-1,  0,  1,  0,  0,  0],
                      [-1, -1,  0,  1,  0,  0],
                      [ 0, -2,  0,  0,  1,  0],
                      [ 0, -1, -1,  0,  0,  1]])
    >>> K = np.array([50.0, 10.0, 40.0, 100.0])
    >>> c0 = np.zeros((3, 6))
    >>> c0[:,0] = 0.001  # x_A0 = 0.001
    >>> c0[:,1] = np.array([0.0, 0.0005, 0.001]) # x_B0, only few for display
    >>> c = eqtk.sweep_titration(N, K, c0, units='mM')
    >>> c
        array([[  1.96078431e-05,   0.00000000e+00,   9.80392157e-04,
                  0.00000000e+00,   0.00000000e+00,   0.00000000e+00],
               [  1.87907443e-05,   4.42652649e-04,   9.39537214e-04,
                  8.31777274e-08,   7.83765472e-06,   4.15888637e-05],
               [  1.80764425e-05,   8.62399860e-04,   9.03822124e-04,
                  1.55891215e-07,   2.97493407e-05,   7.79456073e-05]])

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

    >>> import numpy as np
    >>> import eqtk
    >>> A = np.array([[ 1,  0,  0,  1,  1],
                      [ 0,  1,  0,  1,  0],
                      [ 0,  0,  1,  0,  1]])
    >>> G = np.array([0.0, 1.0, -2.0, -3.0, -7.0])
    >>> c0 = np.array([0.005, 0.001, 0.002])
    >>> x, run_stats = eqtk.sweep_titration(None, None, c0, A=A, G=G, units=None,
                                return_run_stats=True)
    >>> x
    array([ 0.00406569,  0.00081834,  0.00124735,  0.00018166,  0.00075265])
    >>> run_stats.__dict__
    {'n_cauchy_steps': 0,
     'n_chol_fail_cauchy_steps': 0,
     'n_dogleg_fail': 0,
     'n_dogleg_steps': 0,
     'n_irrel_chol_fail': 0,
     'n_newton_steps': 11}

    4) Competition assay.  Species A binds both B and C with specified
       dissociation constants, Kd.
           AB <--> A + B,      Kd = 50 nM
           AC <--> A + C       Kd = 10 nM
       Initial concentrations of [A]_0 = 2.0 uM, [B]_0 = 0.05 uM,
       [C]_0 = 1.0 uM.  The ordering of the compounds in the example
       is A, B, C, AB, AC.

    >>> import numpy as np
    >>> import eqtk
    >>> N = np.array([[ 1,  1,  0, -1,  0],
                      [ 1,  0,  1,  0, -1]])
    >>> K = np.array([0.05, 0.01])
    >>> c0 = np.array([2.0, 0.05, 1.0, 0.0, 0.0])
    >>> c = eqtk.sweep_titration(N, K, c0, units='uM')
    >>> c
    array([ 0.96274868,  0.00246853,  0.01028015,  0.04753147,  0.98971985])
    """
    single_point = False
    if len(c0.shape) == 1:
        single_point = True

    c0, N, K, A, G = checks.check_input(c0, N, K, A, G)

    # Solve for mole fractions
    if G is None:
        x0, K, solvent_density = _nondimensionalize_NK(
            c0, N, K, T, solvent_density, units
        )

        x = _solve_NK(
            N,
            -np.log(K),
            x0,
            max_iters=max_iters,
            tol=tol,
            delta_bar=1000.0,
            eta=eta,
            min_delta=min_delta,
            max_trials=max_trials,
            perturb_scale=perturb_scale,
        )
    else:
        x0, G, solvent_density = _nondimensionalize_AG(
            c0, G, T, solvent_density, units, G_units
        )

        x = _solve_AG(
            A,
            G,
            x0,
            max_iters=max_iters,
            tol=tol,
            delta_bar=1000.0,
            eta=eta,
            min_delta=min_delta,
            max_trials=max_trials,
            perturb_scale=perturb_scale,
        )

    # Convert x to appropriate units
    x *= solvent_density

    # If a single titration point was inputted, return single 1D array
    if single_point:
        return x.flatten()
    return x


def volumetric_titration(
    c0,
    initial_volume,
    c0_titrant,
    vol_titrated,
    N=None,
    K=None,
    A=None,
    G=None,
    solvent_density=None,
    units=None,
    T=20.0,
    G_units=None,
    inputs_guaranteed=False,
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
    solution for points along a titration curve.

    Parameters
    ----------
    N : array_like, shape (n_reactions, n_compounds)
        N[r][j] = the stoichiometric coefficient of compound j
        in chemical reaction r.  Ignored if G is not None.
    K : array_like, shape (n_reactions,)
        K[r] is the equilibrium constant for chemical reaction r.
        Ignored if G is not None.
    c0 : array_like, shape (n_compounds,)
        Array containing the total "initial" concentration of all compounds
        in solution before equilibrating and before any of the titrant
        solution is added.
    initial_volume : float
        Initial volume of the solution to which the titrant is added.
        Must of the same units as the entries in vol_titrated.
    c0_titrant : array_like, shape (n_compounds,)
        The concentration of each species in the titrant solution.
    vol_titrated : array_like, shape (n_titration_points,)
        Array containing volume of titrant solution to add.
    solvent_density : float, default = None
        The density of the solvent in units commensurate with the units
        keyword argument.  Default (None) assumes the solvent is water,
        and its density is computed at the temperature specified by the
        T keyword argument.
    units : string, default = 'molar'
        The units of the given concentrations.  Allowable values are
        'molar', 'M', 'millimolar', 'mM', 'micromolar', 'uM',
        'nanomolar', 'nM', 'picomolar', 'pM', None.  If None, concen-
        trations are given as mole fractions.  The equilbrium constants
        given by K have corresponding units.  The output is also given
        in these units.
    T : float, default = 20.0
        Temperature, in deg. C, of the solution.  Not relevant when
        units and G_units is None
    A : array_like, shape (n_constraints, n_compounds)
        Constraint matrix. No column may be repeated.  Ignored if
        G is None.
    G : array_like, shape (n_compounds, ), default None
        G[j] is the free energy, either in units of kcal/mol or in units
        of kT, depending on the value of G_units.  If not None, A must also
        be not None.  If this is the case overrides any input for N and K.
    G_units : string, default 'kcal/mol'
        Units in which free energy is given.  If None, the free energies are
        specified in units of kT.  Other acceptable options are: 'kcal/mol',
        'J', 'J/mol', 'kJ/mol', and 'pN-nm'.
    return_run_stats : Boolean, default = False
        If True, also returns a list of statistics on steps taken by
        the trust region algorithm.
    inputs_guaranteed : Boolean, default = False
        If True, skips input checking.  This is used if speed is impor-
        tant, e.g., if the function is called repeatedly.
    max_iters : int, default = 1000
        Maximum number of interations allowed in trust region
        method.
    tol : float, default = 0.0000001
        Tolerance for convergence.  The absolute tolerance is
        tol * (mininium single-species initial mole fraction)
    delta_bar : float, default = 1000.0
        Maximum step size allowed in the trust region method.
    eta : float, default = 0.125
        Value for eta in the trust region method (see Nocedal and
        Wright reference). 0 < eta < 1/4.
    min_delta : float, default = 1e-12
        Minimal allowed radius of the trust region.  When the trust region
        radius gets below min_delta, the trust region iterations stop,
        and a final Newton step is attempted.
    write_log_file : Boolean, default = False
        True to write information about trust region calculation to a
        log file.
    log_file : string, default = 'conc_calc.log'
        Name of file for printing information about trust region
        calculation.

    Returns
    -------
    c : array_like, shape (n_titration_points, n_compounds)
        c[k,j] = the equilibrium concentration of compound when the
        concentration of the titrated species is c0_titrated[k].
        Units are given as specified in the units keyword argument.

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

    .. All K's must be positive and finite.

    .. All c0's must be nonnegative and finite.

    .. If N and A are specified,

    References
    ----------
    .. [1] J.S. Bois, Analysis of nucleic acids in dilute solutions,
           Caltech Ph.D. thesis, 2007.

    Examples
    --------
    2) Compute the pH titration curve of a 1 M solution of weak acid HA
       with acid dissociation constant 1e-5 M, titrating in 1.0 M NaOH.

    >>> import readout as ro
    >>> import eqtk
    >>> N = np.array([[1,  1,  0,  0],
                      [1,  0, -1,  1]])
    >>> K = np.array([1.0e-14, 1.0e-5])
    >>> c0 = np.array([1.0e-7, 1.0e-7, 1.0, 0.0])
    >>> c0_titrant = np.array([0.0, 1.0, 0.0, 0.0])
    >>> initial_volume = 0.1
    >>> vol_titrated = np.array([0.0, 0.1, 0.2]) # Only a few for display
    >>> c = eqtk.volumetric_titration(N, K, c0, initial_volume, c0_titrant,
                                    vol_titrated, units='M')
    >>> c
        array([[  3.15728161e-03,   3.16728162e-12,   9.96842719e-01,
                  3.15728162e-03],
               [  4.47219126e-10,   2.23604032e-05,   2.23599563e-05,
                  4.99977640e-01],
               [  2.99999999e-14,   3.33333334e-01,   9.99999994e-10,
                  3.33333332e-01]])
    """

    if initial_volume <= 0:
        raise ValueError("initial volume must be > 0")
    if any(c0_titrant < 0):
        raise ValueError("c0_titrant must have non-negative concentrations.")
    if any(vol_titrated < 0):
        raise ValueError("vol_titrated must have non-negative volumes.")

    new_c0 = volumetric_to_c0(c0, c0_titrant, initial_volume, vol_titrated)

    c = solve(
        c0=new_c0,
        N=N,
        K=K,
        units=units,
        solvent_density=solvent_density,
        T=T,
        A=A,
        G=G,
        G_units=G_units,
        max_iters=1000,
        tol=0.0000001,
        delta_bar=1000.0,
        eta=0.125,
        min_delta=1.0e-12,
        max_trials=100,
        perturb_scale=100.0,
    )

    return c


def final_value_titration(
    x0,
    final_x,
    final_species,
    N,
    K,
    final_tolerance=1e-6,
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
    solution given the dimensionless equilibrium constants for the
    possible chemical reactions. Attempt to solve for each point in
    a curve where the desired concentration of a single species is
    specified for each point by specified_x. The particular species
    is specified by specified_species. Each point is found by
    Ridder's method for now. Can use Newton's method if we find an
    exact derivative later.

    Parameters
    ----------
    N : array_like, shape (n_reactions , n_compounds)
        N[r][j] = the stoichiometric coefficient of compounds j
        in chemical reaction r.
    K : array_like, shape (n_reactions,)
        K[r] is the equilibrium constant for chemical reaction r
    x0 : array_like, shape (n_compounds,)
        Array containing the total "initial" mole fraction of all
        compounds in solution.
    final_x : array_like, shape (n_points,)
        Array containing the total "final" mole fraction of a
        particular compound in solution.
    final_species : Integer
        The index of the species referred to by final_x
    final_tolerance : Float, default=1e-14
        The relative tolerance of the final concentration with
        respect to final_x

    Returns
    -------
    x : array_like, shape (n_points, n_compounds)
        x[i, j] = the equilibrium mole fraction of compound j at point i
    converged : Boolean
        True is trust region calculation converged, False otherwise.

    Raises
    ------
    ValueError
        If N, K, and x0 have a dimensional mismatch.

    """
    n_reactions, n_compounds = N.shape

    # Add a new reaction: just straight up production of the final_species
    N_new = np.vstack((np.zeros(n_compounds), N))
    N_new[0, final_species] = 1
    K_new = np.concatenate(((0,), K))

    result_x = np.empty((len(final_x), n_compounds))

    for i, pt in enumerate(final_x):
        # Set concentration of final_species by setting K of its prod. rxn.
        K_new[0] = pt

        res_x, converged, res_stats = solve(
            N_new,
            K_new,
            x0,
            max_iters=max_iters,
            tol=tol,
            delta_bar=1000.0,
            eta=eta,
            min_delta=min_delta,
            max_trials=max_trials,
            perturb_scale=perturb_scale,
        )
        result_x[i, :] = res_x

    return result_x


def volumetric_to_c0(c0, c0_titrant, initial_volume, vol_titrated):
    """Convert volumetric titration to input concentrations."""
    n_titration_points = len(vol_titrated)
    n_compounds = len(c0_titrant)
    if n_compounds != len(c0):
        raise ValueError("Dimensions of c0 and c0_titrant must agree")

    vol_titrated_col = np.reshape(vol_titrated, (n_titration_points, 1))
    c_titrant_row = np.reshape(c0_titrant, (1, n_compounds))
    c0_row = np.reshape(c0, (1, n_compounds))
    titrated_scale = vol_titrated_col / (initial_volume + vol_titrated_col)
    original_scale = initial_volume / (initial_volume + vol_titrated_col)

    return np.dot(titrated_scale, c_titrant_row) + np.dot(original_scale, c0_row)


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
    return_run_stats=False,
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


def _prune_AG(A, G, x0, A_positive):
    """Prune constraint matrix and free energy to ignore inert and
    missing species.
    """
    constraint_vector = np.dot(A, x0)

    if A_positive:
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
    else:
        N = linalg.nullspace_svd(A).transpose()
        dummy_minus_log_K = np.ones(N.shape[0])
        N_new, dummy_K_new, x0_new, active_compounds, _ = _prune_NK(
            N, dummy_minus_log_K, x0
        )
        A_new_F = linalg.nullspace_svd(N).transpose()
        A_new = np.ascontiguousarray(A_new_F)
        constraint_vector_new = np.dot(A_new, x0_new)

    G_new = _boolean_index(G, active_compounds, np.sum(active_compounds))

    return A_new, G_new, constraint_vector_new, active_compounds


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


def _solve_NK(
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
    ret_shape = x0.shape

    n_titration_points = x0.shape[0]

    x = np.empty((n_titration_points, n_compounds))
    for i_point in range(n_titration_points):
        N_new, minus_log_K_new, x0_new, active_compounds, _ = _prune_NK(
            N, minus_log_K, x0[i_point]
        )
        if len(minus_log_K_new) > 0:
            n_reactions_new, n_compounds_new = N_new.shape
            n_constraints_new = n_compounds_new - n_reactions_new

            # Compute and check stoichiometric matrix
            A = np.ascontiguousarray(linalg.nullspace_svd(N_new).transpose())

            # If completely constrained (N square), solve directly
            if n_constraints_new == 0:
                ln_x = np.linalg.solve(N_new, -minus_log_K_new)
                x_new = np.exp(ln_x)
            else:
                # Compute the free energies in units of kT from the K's
                b = np.concatenate((np.zeros(n_constraints_new), minus_log_K_new))
                N_prime = np.vstack((A, N_new))
                G = np.linalg.solve(N_prime, b)
                constraint_vector = np.dot(A, x0_new)

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

    x = np.reshape(x, ret_shape)

    return x


def _solve_AG(
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
        be the identitity matrix.  No column may be repeated.
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

    n_titration_points = x0.shape[0]

    x = np.empty((n_titration_points, n_compounds))

    if np.all(A >= 0):
        A_nonnegative = True
    else:
        A_nonnegative = False

    for i_point in range(n_titration_points):
        A_new, G_new, constraint_vector, active_compounds = _prune_AG(
            A, G, x0[i_point], A_nonnegative
        )

        # Detect if A is empty (no constraints)
        A_empty = (A_new.shape[0] + A_new.shape[1] == 1)

        # Problem is entirely contrained, must have x = x0.
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


def _nondimensionalize_NK(c0, N, K, T, solvent_density, units):
    # Compute solvent density in appropriate units
    solvent_density = _parse_solvent_density(solvent_density, T, units)

    # Convert K's and c0 to dimensionless
    K_nondim = K / solvent_density ** N.sum(axis=1)
    c0_nondim = _nondimensionalize_c0(c0, solvent_density)

    return c0_nondim, K_nondim, solvent_density


def _nondimensionalize_AG(c0, G, T, solvent_density, units, G_units):
    # Compute solvent density in appropriate units
    solvent_density = _parse_solvent_density(solvent_density, T, units)

    # Convert G's and c0 to dimensionless
    G_nondim = _dimensionless_free_energy(G, G_units, T)
    c0_nondim = _nondimensionalize_c0(c0, solvent_density)

    return c0_nondim, G_nondim, solvent_density


def _parse_solvent_density(solvent_density, T, units):
    if solvent_density is None:
        return _water_density(T, units)
    elif (units is None or units == "") and solvent_density != 1.0:
        raise ValueError(
            "If `solvent_density` is specified, `units` must also be specified."
        )

    return solvent_density


def _nondimensionalize_c0(c0, solvent_density):
    """Nondimensionalize input concentration by solvent density."""
    return c0 / solvent_density


def _water_density(T, units):
    """
    Calculate the number density of water in specified units.

    Parameters
    ----------
    T : float
        Temperature in degrees C.
    units : string, default = 'M'
        The units in which the density is to be calculated.
        Valid values are: 'M', 'mM', 'uM', 'nM', 'pM'.

    Returns
    -------
    water_density : float
        Number of moles of water per liter.

    References
    ----------
    Tanaka M., Girard, G., Davis, R., Peuto A.,
    Bignell, N.   Recommended table for the denisty
    of water..., Metrologia, 2001, 38, 301-309
    """
    # If dimensionless, take solvent density to be unity
    if units is None or units == "":
        return 1.0

    a1 = -3.983035
    a2 = 301.797
    a3 = 522528.9
    a4 = 69.34881
    a5 = 999.974950

    # Compute water density in units of molar
    dens = a5 * (1 - (T + a1) * (T + a1) * (T + a2) / a3 / (T + a4)) / 18.0152

    # Valid units
    allowed_units = [None, "M", "mM", "uM", "nM", "pM"]

    # Convert to specified units
    if units in ["millimolar", "mM"]:
        dens *= 1000.0
    elif units in ["micromolar", "uM", "µM"]:
        dens *= 1000000.0
    elif units in ["nanomolar", "nM"]:
        dens *= 1000000000.0
    elif units in ["picomolar", "pM"]:
        dens *= 1000000000000.0
    elif units not in ["molar", "M"]:
        raise ValueError(
            f"Specified concentration units of {units} not in {allowed_units}."
        )

    return dens


def _dimensionless_free_energy(G, units, T=None):
    """
    Convert free energy to dimensionless units, where G is in given units.
    """
    if units is None or units == "kT":
        return G
    elif T is None:
        raise ValueError("If G is specified with units, must also supply T.")

    kT = _thermal_energy(T, units)
    return G / kT


def _thermal_energy(T, units):
    """
    Return value of thermal energy kT in specified units. T is
    assumed to be in deg. C.
    """
    if T > 150.0:
        warnings.warn("WARNING: T may be in wrong units, must be in deg. C.")

    T += constants.absolute_zero

    allowed_units = ["kcal/mol", "J", "J/mol", "kJ/mol", "pN-nm"]

    if units == "kcal/mol":
        return constants.kB_kcal_per_mol * T
    elif units == "J":
        return constants.kB_J * T
    elif units == "J/mol":
        return constants.kB_J_per_mol * T
    elif units == "kJ/mol":
        return constants.kB_kJ_per_mol * T
    elif units == "pN-nm":
        return constants.kB_pN_nm * T
    else:
        raise ValueError(
            f"Specified thermal energy units of {units} not in {allowed_units}."
        )


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


# Use Numba'd functions
if numba_check.numba_check():
    import numba

    _boolean_index = numba.jit(_boolean_index, nopython=True)
    _boolean_index_2d = numba.jit(_boolean_index_2d, nopython=True)
    _print_runstats = numba.jit(_print_runstats, nopython=True)
    _initial_guess = numba.jit(_initial_guess, nopython=True)
    _perturb_initial_guess = numba.jit(_perturb_initial_guess, nopython=True)
    _prune_NK = numba.jit(_prune_NK, nopython=True)
    _prune_AG = numba.jit(_prune_AG, nopython=True)
    _solve_trust_region = numba.jit(_solve_trust_region, nopython=True)
    _solve_NK = numba.jit(_solve_NK, nopython=True)
    _solve_AG = numba.jit(_solve_AG, nopython=True)
