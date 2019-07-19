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


def conc(
    c0,
    N=None,
    K=None,
    A=None,
    G=None,
    units=None,
    solvent_density=None,
    T=20.0,
    G_units=None,
    quiet=True,
    return_run_stats=False,
    inputs_guaranteed=False,
    **kwargs
):
    """
    Solve for equilibrium concentrations of all species in a dilute
    solution.

    Parameters
    ----------
    c0 : array_like, shape (n_compounds, )
        Array containing the total "initial" concentration of all compounds
        in solution for each titration point.  c0[i,j] is the initial
        concentration of compound j at titration point i.
    N : array_like, shape (n_reactions, n_compounds)
        N[r,j] = the stoichiometric coefficient of compound j
        in chemical reaction r.  Ignored if G is not None.
    K : array_like, shape (n_reactions,)
        K[r] is the equilibrium constant for chemical reaction r.
        Ignored if G is not None.
    A : array_like, shape (n_constraints, n_compounds)
        Constraint matrix. All rows must be linearly independent.
        Ignored if G is None.
    G : array_like, shape (n_compounds, ), default None
        G[j] is the free energy, either in units of kcal/mol or in units
        of kT, depending on the value of G_units.  If not None, A must also
        be not None.  If this is the case overrides any input for N and K.
    units : string, default = 'molar'
        The units of the given concentrations.  Allowable values are
        'molar', 'M', 'millimolar', 'mM', 'micromolar', 'uM',
        'nanomolar', 'nM', 'picomolar', 'pM', None.  If None, concen-
        trations are given as mole fractions.  The equilbrium constants
        given by K have corresponding units.  The output is also given
        in these units.
    solvent_density : float, default = None
        The density of the solvent in units commensurate with the units
        keyword argument.  Default (None) assumes the solvent is water,
        and its density is computed at the temperature specified by the
        T keyword argument.
    T : float, default = 20.0
        Temperature, in deg. C, of the solution.  Not relevant when
        units and G_units is None,
    G_units : string, default 'kcal/mol'
        Units in which free energy is given.  If None, the free energies are
        specified in units of kT.  Other acceptable options are: 'kcal/mol',
        'J', 'J/mol', 'kJ/mol', and 'pN-nm'.
    quiet : Boolean, default = True
        True to surpress output to the screen.
    return_run_stats : Boolean, default = False
        If True, also returns a list of statistics on steps taken by
        the trust region algorithm.
    inputs_guaranteed : Boolean, default = False
        If True, skips input checking.  This is used if speed is impor-
        tant, e.g., if the function is called repeatedly.

    Optional algorithmic key word arguments
    ---------------------------------------
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
    max_trial : int, default = 40
        Maximum number of initial guesses to be tried.  In practice, we
        have never seen more than one initial trial necessary for
        convergence.  In theory, only one trial is necessary with arbitrary
        precision.
    perturb_scale : float, default = 100.0
        Multiplier on random perturbations to the initial guesses
        as new ones are generated.
    seed : int, default = 0
        Positive seed for random number generation.  If seed = 0,
        random numbers are seeded off the current time.

    Returns
    -------
    c : array_like, shape (n_compounds, )
        c[j] = the equilibrium concentration of compound j.
        Units are given as specified in the units keyword argument.
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
    .. `conc` equivalent to `sweep_titration`.

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
    >>> c0 = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    >>> c = eqtk.conc(c0, N, K, units='mM')
    >>> c
    array([ 0.00121271,  0.15441164,  0.06063529,  0.00187256,  0.95371818,
            0.93627945])


    2) Find the equilibrium concentrations of a solution containing
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
    >>> c0 = np.array([0.005, 0.001, 0.002, 0.0, 0.0])
    >>> x, run_stats = eqtk.conc(c0, A=A, G=G, units=None, 
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

    3) Competition assay.  Species A binds both B and C with specified
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
    >>> c = eqtk.conc(N, K, c0, units='uM')
    >>> c
    array([ 0.96274868,  0.00246853,  0.01028015,  0.04753147,  0.98971985])
    """
    return sweep_titration(
        c0,
        N=N,
        K=K,
        A=A,
        G=G,
        units=units,
        solvent_density=solvent_density,
        T=T,
        G_units=G_units,
        quiet=quiet,
        return_run_stats=return_run_stats,
        inputs_guaranteed=inputs_guaranteed,
        **kwargs
    )


def sweep_titration(
    c0,
    N=None,
    K=None,
    A=None,
    G=None,
    units=None,
    solvent_density=None,
    T=20.0,
    G_units=None,
    quiet=True,
    return_run_stats=False,
    inputs_guaranteed=False,
    **kwargs
):
    """
    Solve for equilibrium concentrations of all species in a dilute
    solution.

    Parameters
    ----------
    c0 : array_like, shape (n_titration_points, n_compounds)
        Array containing the total "initial" concentration of all compounds
        in solution for each titration point. c0[i,j] is the initial
        concentration of compound j at titration point i.
    N : array_like, shape (n_reactions, n_compounds)
        N[r,j] = the stoichiometric coefficient of compound j
        in chemical reaction r.  Ignored if G is not None.
    K : array_like, shape (n_reactions,)
        K[r] is the equilibrium constant for chemical reaction r.
        Ignored if G is not None.
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
    A : array_like, shape (n_constraints, n_compounds)
        Constraint matrix. All rows must be linearly independent.
        Ignored if G is None.
    G : array_like, shape (n_compounds, ), default None
        G[j] is the free energy, either in units kT or in units specified
        by G_units.  If not None, A must also be not None.  If this is
        the case, overrides any input for N and K.
    G_units : string, default None
        Units in which free energy is given.  If None, the free energies are
        specified in units of kT.  Other acceptable options are: 'kcal/mol',
        'J', 'J/mol', 'kJ/mol', and 'pN-nm'.
    quiet : Boolean, default = True
        True to surpress output to the screen.
    return_run_stats : Boolean, default = False
        If True, also returns a list of statistics on steps taken by
        the trust region algorithm.
    inputs_guaranteed : Boolean, default = False
        If True, skips input checking.  This is used if speed is impor-
        tant, e.g., if the function is called repeatedly.

    Optional algorithmic key word arguments
    ---------------------------------------
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
    max_trial : int, default = 40
        Maximum number of initial guesses to be tried.  In practice, we
        have never seen more than one initial trial necessary for
        convergence.  In theory, only one trial is necessary with arbitrary
        precision.
    perturb_scale : float, default = 100.0
        Multiplier on random perturbations to the initial guesses
        as new ones are generated.
    seed : int, default = 0
        Positive seed for random number generation.  If seed = 0,
        random numbers are seeded off the current time.

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

    # Pop the trust region kwargs and assign defaults
    trust_region_params = TrustRegionParams(**kwargs)

    # Check inputs
    if not inputs_guaranteed:
        c0, N, K, A, G = checks.check_eq_input(c0, N, K, A, G)

    # Solve for mole fractions
    if G is None:
        c0, K, solvent_density = _nondimensionalize_NK(
            c0, N, K, T, solvent_density, units
        )

        x, run_stats = eqtk_conc_pure_python(
            N, -np.log(K), c0, trust_region_params=trust_region_params, quiet=quiet
        )
    else:
        c0, G, solvent_density = _nondimensionalize_AG(
            c0, G, T, solvent_density, units, G_units
        )
        x, run_stats = eqtk_conc_from_free_energies_pure_python(
            A, G, c0, trust_region_params=trust_region_params, quiet=quiet
        )

    # Convert x to appropriate units
    x *= solvent_density

    # Return values
    if return_run_stats:
        return x, run_stats
    return x


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
    quiet=True,
    return_run_stats=False,
    inputs_guaranteed=False,
    **kwargs
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
    quiet : Boolean, default = True
        True to surpress output to the screen.
    return_run_stats : Boolean, default = False
        If True, also returns a list of statistics on steps taken by
        the trust region algorithm.
    inputs_guaranteed : Boolean, default = False
        If True, skips input checking.  This is used if speed is impor-
        tant, e.g., if the function is called repeatedly.

    Optional algorithmic key word arguments
    ---------------------------------------
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
    max_trial : int, default = 40
        Maximum number of initial guesses to be tried.  In practice, we
        have never seen more than one initial trial necessary for
        convergence.  In theory, only one trial is necessary with arbitrary
        precision.
    perturb_scale : float, default = 100.0
        Multiplier on random perturbations to the initial guesses
        as new ones are generated.
    seed : int, default = 0
        Positive seed for random number generation.  If seed = 0,
        random numbers are seeded off the current time.
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

    c, run_stats = sweep_titration(
        c0=new_c0,
        N=N,
        K=K,
        units=units,
        solvent_density=solvent_density,
        T=T,
        A=A,
        G=G,
        G_units=G_units,
        quiet=quiet,
        return_run_stats=True,
        inputs_guaranteed=inputs_guaranteed,
        **kwargs
    )

    if return_run_stats:
        return c, run_stats
    return c


def final_value_titration(
    c0,
    set_species,
    set_cf,
    N,
    K,
    solvent_density=None,
    units=None,
    T=20.0,
    quiet=True,
    return_run_stats=False,
    inputs_guaranteed=False,
    **kwargs
):
    """
    Solve for the equilibrium concentrations, setting a subset of the
    final concentrations to known values. Some of the theory still
    needs to be finished up for this functionality. It has some
    serious inefficiencies left.

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
    set_species: array_like, shape (n_set, )
        An array containing the indices of the species to set the final
        concentrations of.
    set_c_f : array_like, shape (n_set, n_titration_points)
        A 2-D array containing the desired final concentration of each
        species in set_species, for each final titration point.
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
    quiet : Boolean, default = True
        True to surpress output to the screen.
    return_run_stats : Boolean, default = False
        If True, also returns a list of statistics on steps taken by
        the trust region algorithm.
    inputs_guaranteed : Boolean, default = False
        If True, skips input checking.  This is used if speed is impor-
        tant, e.g., if the function is called repeatedly.

    Optional algorithmic key word arguments
    ---------------------------------------
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
    max_trial : int, default = 40
        Maximum number of initial guesses to be tried.  In practice, we
        have never seen more than one initial trial necessary for
        convergence.  In theory, only one trial is necessary with arbitrary
        precision.
    perturb_scale : float, default = 100.0
        Multiplier on random perturbations to the initial guesses
        as new ones are generated.
    seed : int, default = 0
        Positive seed for random number generation.  If seed = 0,
        random numbers are seeded off the current time.
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

    Examples
    --------
    2) Compute the pH titration curve of a 1 M solution of weak acid HA
       with acid dissociation constant 1e-5 M, titrating in 1.0 M NaOH.
    """

    trust_region_params = TrustRegionParams(**kwargs)

    if not inputs_guaranteed:
        c0, N, K, A, G = checks.check_eq_input(c0, N, K)

    n_compounds = len(c0)

    set_species_array = np.array(set_species)
    if len(set_species_array.shape) == 0:
        set_species_array = np.array([set_species])
    elif len(set_species_array.shape) > 1:
        raise ValueError("Set species must be zero or 1 dimensional")
    elif any(set_species_array >= n_compounds):
        raise ValueError("Set species must be in c0")

    n_set_species = set_species_array.shape[0]

    titration_points = np.array(set_c_f)
    if len(titration_points.shape) == 0:
        titration_points = np.array([[set_c_f]])
    elif len(titration_points.shape) == 1:
        titration_points = np.array([set_c_f]).transpose()
    elif len(titration_points.shape) > 2:
        raise ValueError("set_c_f must have at most 2 dimensions")

    if titration_points.shape[1] != n_set_species:
        raise ValueError("set_species and set_c_f dimensions don't agree")

    n_titration_points = titration_points.shape[0]

    if N.shape[1] != n_compounds:
        raise ValueError("c0 and N dimensions must agree")

    top_N = np.zeros((n_set_species, n_compounds))
    top_N[range(n_set_species), set_species] = 1
    new_N = np.vstack((top_N, N))
    n_reacs_full = new_N.shape[0]

    if n_reacs_full > n_compounds:
        raise ValueError(
            "The set of reactions and set concentrations" + "is overdetermined"
        )

    u, s, v = np.linalg.svd(new_N)
    approx_rank = (s > 1e-10).sum()
    if approx_rank < n_reacs_full:
        raise ValueError(
            "The set of reactions and set concentrations"
            + "are not linearly independent"
        )

    if units is None or units == "":
        if solvent_density is not None and solvent_density != 1.0:
            raise ValueError("If solvent density is specified, so must units.")
        solvent_density = 1.0
    elif solvent_density is None:
        solvent_density = _water_density(T, units)

    K_nondim = K / solvent_density ** N.sum(1)
    titration_points_nondim = titration_points / solvent_density
    c0_nondim = c0 / solvent_density

    c = np.zeros((n_titration_points, n_compounds))

    for i, c_f in enumerate(titration_points_nondim):
        new_K = np.concatenate([c_f, K_nondim], 0)
        try:
            x, converged, run_stats = eqtk_conc(
                new_N,
                -np.log(new_K),
                c0_nondim,
                trust_region_params=trust_region_params,
                quiet=quiet,
            )
        except RuntimeError:
            try:
                x, run_stats = eqtk_conc_pure_python(
                    new_N,
                    -np.log(new_K),
                    c0_nondim,
                    trust_region_params=trust_region_params,
                    quiet=quiet,
                )
                converged = True
            except:
                converged = False
        if not converged:
            raise RuntimeError("Calculation of concentrations did not converge!")

        c[i, :] = x * solvent_density

    if return_run_stats:
        return c, run_stats
    return c


def eqtk_conc_optimize_pure_python(
    A, G, constraint_vector, trust_region_params=None, quiet=True
):
    """
    Solve for equilibrium concentrations of all species in a dilute
    solution given their free energies in units of kT.

    Parameters
    ----------
    A : array_like, shape (n_constraints, n_compounds)
        Each row represents a system constaint.
        Namely, np.dot(A, x) = constraint_vector.
    G : array_like, shape (n_compounds,)
        G[j] is the free energy of compound j in units of kT.
    constraint_vector : array_like, shape (n_constraints,)
        Right hand since of constraint equation,
        np.dot(A, x) = constraint_vector.
    trust_region_params : instance of TrustRegionParams class
        Contains pertinent parameters for trust region calculation.
        If None, defaults are used.
    quiet : Boolean, default = True
        True to surpress output to the screen.

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
        n_irrel_chol_fail : # of steps with irrelovant Cholesky failures
        n_dogleg_fail : # of failed dogleg calculations

    Raises
    ------
    ValueError
        If A, G, and constraint_vector have a dimensional mismatch.
    """

    # Fetch trust region parameters if need be
    if trust_region_params is None:
        trust_region_params = TrustRegionParams()

    # Dimension of the problem
    n_constraints, n_compounds = A.shape

    # Build new problem with inactive ones cut out
    params = (G, A, constraint_vector)
    abs_tol = trust_region_params.tol * np.abs(constraint_vector)
    mu0 = get_initial_guess(constraint_vector, G, A, None, None)

    mu, converged, step_tally = trust_region.trust_region_convex_unconstrained(
        mu0,
        G,
        A,
        constraint_vector,
        tol=abs_tol,
        max_iters=trust_region_params.max_iters,
        delta_bar=trust_region_params.delta_bar,
        eta=trust_region_params.eta,
        min_delta=trust_region_params.min_delta,
    )

    # Try other initial guesses if it did not converge
    n_trial = 1
    while not converged and n_trial < trust_region_params.max_trial:
        mu0 = get_initial_guess(
            constraint_vector,
            G,
            A,
            perturb_scale=trust_region_params.perturb_scale,
            mu0=mu0,
        )
        mu, converged, step_tally = trust_region.trust_region_convex_unconstrained(
            mu0,
            G,
            A,
            constraint_vector,
            tol=abs_tol,
            max_iters=trust_region_params.max_iters,
            delta_bar=trust_region_params.delta_bar,
            eta=trust_region_params.eta,
            min_delta=trust_region_params.min_delta,
        )
        n_trial += 1

    run_stats = dict()
    run_stats["max_n_trials"] = trust_region_params.max_trial
    run_stats["n_trials"] = n_trial
    run_stats["n_constraints"] = n_constraints
    run_stats["n_iterations"] = int(np.sum(step_tally))
    run_stats["n_newton_steps"] = int(step_tally[0])
    run_stats["n_cauchy_steps"] = int(step_tally[1])
    run_stats["n_dogleg_steps"] = int(step_tally[2])
    run_stats["n_chol_fail_cauchy_steps"] = int(step_tally[3])
    run_stats["n_irrel_chol_fail"] = int(step_tally[4])
    run_stats["n_dogleg_fail"] = int(step_tally[5])

    x = np.exp(trust_region.compute_logx(mu, G, A))

    return x, converged, run_stats


def get_initial_guess(constraint_vector, G, A, perturb_scale=None, mu0=None):
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

    # Guess mu such that ln x = 1 for all x (x ~ 3).
    mu_guess = np.linalg.solve(np.dot(A, A.transpose()), np.dot(A, G + 1.0))

    # OLD WAY
    # mu_guess = ((1.0 + G) / abs(A).sum(0)).min() \
    #    * ones_like(constraint_vector)

    # Perturb if desired
    if mu0 is not None:
        new_mu = mu0 + perturb_scale * 2.0 * (
            np.random.rand(len(constraint_vector)) - 0.5
        )
        while (-G + np.dot(new_mu, A)).max > 250.0:  # Prevent overflow err
            perturb_scale /= 2.0
    else:
        new_mu = mu_guess

    return new_mu


def prune_NK(N, minus_log_K, x0):
    """Prune reactions to ignore inert and missing species.
    """
    n_reactions, n_compounds = N.shape

    prev_active = x0 > 0
    active_compounds = x0 > 0
    active_reactions = np.zeros(n_reactions, dtype=bool)
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
        prev_active = np.array(active_compounds)

    # Select all compounds that are in at least one active reaction
    active_compounds = np.dot(active_reactions, N != 0)

    n_reactions_new = np.dot(np.ones(n_reactions), active_reactions)
    n_compounds_new = np.dot(np.ones(n_compounds), active_compounds)
    if n_reactions_new > 0 and n_compounds_new > 0:
        minus_log_K_new = minus_log_K[active_reactions]
        N_new = N[active_reactions, :]
        N_new = N_new[:, active_compounds]
        x0_new = x0[active_compounds]
    else:
        N_new = np.array([[]])
        minus_log_K_new = np.array([])
        x0_new = x0

    return N_new, minus_log_K_new, x0_new, active_compounds, active_reactions


def prune_AG(A, G, x0, elemental):
    """Prune constraint matrix and free energy to ignore inert and
    missing species.

    The matrix A must be elemental.
    """

    # Generate new x0 for only particles
    x0_part = np.dot(A, x0)

    if not elemental:
        return A, G, x0_part, np.array([True] * len(x0))

    # Find active particles/compounds
    active_particles = x0_part > 0

    done = False
    while not done:
        # Determine which compounds are active (there are no inactive
        # particles contained in them).
        prev_active = np.array(active_particles)
        active_compounds = np.dot(active_particles == False, A > 0) == False

        # Determine which particles are active
        # There is more than one sink for them
        num_active = np.dot(np.array(A > 0, dtype=int), active_compounds)
        active_particles = num_active > 1

        done = np.all(prev_active == active_particles)

    # Return the original concentrations if there is no freedom
    if active_particles.sum() > 0:
        x0_new = x0_part[np.nonzero(active_particles == 1)[0]]
        A_new_1 = A[np.nonzero(active_particles)[0], :]
        A_new = A_new_1[:, np.nonzero(active_compounds)[0]]
        G_new = G[np.nonzero(active_compounds)[0]]
    else:
        A_new = np.array([[]])
        G_new = np.array([])
        x0_new = x0

    return np.ascontiguousarray(A_new, dtype='float'), G_new, x0_new, active_compounds


def eqtk_conc_pure_python(N, minus_log_K, x0, trust_region_params=None, quiet=True):
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
    trust_region_params : instance of TrustRegionParams class
        Contains pertinent parameters for trust region calculation.
        If None, defaults are used.
    quiet : Boolean, default = True
        True to suppress output to the screen.

    Returns
    -------
    x : array_like, shape (n_compounds)
        x[j] = the equilibrium concentration of compound j.  Units are
        given as specified in the units keyword argument.
    run_stats : a class with atrributes:
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
        If input is in any way invalid.
    RuntimeError
        If the trust region algorithm failed to converge.

    Notes
    -----
    .. N must have full row rank, i.e., all rows must be
       linearly independent.

    .. All x0's must be non-negative and finite

    """

    # Fetch trust region parameters if need be
    if trust_region_params is None:
        trust_region_params = TrustRegionParams()

    # Get number of particles and compounds
    n_reactions, n_compounds = N.shape
    ret_shape = x0.shape

    if len(x0.shape) == 1:
        x0 = np.reshape(x0, (1, len(x0)), order="C")

    n_titration_points = x0.shape[0]

    # Check argument lengths
    if x0.shape[1] != n_compounds:
        raise ValueError("x0 must contain an entry for each compound.")

    x = np.empty((n_titration_points, n_compounds))
    for i_point in range(n_titration_points):
        N_new, minus_log_K_new, x0_new, active_compounds, _ = prune_NK(
            N, minus_log_K, x0[i_point]
        )
        if len(minus_log_K_new) > 0:
            n_reactions_new, n_compounds_new = N_new.shape
            n_constraints_new = n_compounds_new - n_reactions_new

            # Compute and check stoichiometric matrix
            A = linalg.nullspace_svd(N_new).transpose()
            A = np.ascontiguousarray(A, dtype='float')

            if A.shape[0] > n_constraints_new:
                raise ValueError(
                    "Rows in stoichiometric matrix N must " + "be linearly independent."
                )
            elif A.shape[0] < n_constraints_new:
                raise RuntimeError("Improperly specified stoichiometric matrix N.")

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

                x_new, converged, run_stats = eqtk_conc_optimize_pure_python(
                    A,
                    G,
                    constraint_vector,
                    trust_region_params=trust_region_params,
                    quiet=quiet,
                )

                # If not converged, throw exception
                if not converged:
                    err_str = "Calculation of concentrations did not converge!\n"
                    err_str += "Details of calculation:\n"
                    err_str += f"index: {i_point}\n"
                    err_str += "x0:\n" + np.array2string(x0_new, separator=", ") + "\n\n"
                    err_str += "N:\n" + np.array2string(N_new, separator=", ") + "\n\n"
                    err_str += "A:\n" + np.array2string(A, separator=", ") + "\n\n"
                    err_str += "minus_log_K:\n" + np.array2string(minus_log_K_new, separator=", ") + "\n\n"
                    err_str += "G:\n" + np.array2string(G, separator=", ") + "\n\n"
                    err_str += "constraint vector:\n  " + np.array2string(
                        constraint_vector, separator=", "
                    )
                    raise RuntimeError(err_str)

        # Put in concentrations that were cut out
        j = 0
        for i in range(n_compounds):
            if active_compounds[i]:
                x[i_point, i] = x_new[j]
                j += 1
            else:
                x[i_point, i] = x0[i_point, i]

    x = np.reshape(x, ret_shape)

    #### ALSO RETURN RUNSTATS
    return x, None


def eqtk_conc_from_free_energies_pure_python(
    A, G, x0, elemental=False, trust_region_params=None, quiet=True
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
    trust_region_params : instance of TrustRegionParams class
        Contains pertinent parameters for trust region calculation.
        If None, defaults are used.
    quiet : Boolean, default = True
        True to surpress output to the screen.

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

    # Fetch trust region parameters if need be
    if trust_region_params is None:
        trust_region_params = TrustRegionParams()

    n_particles, n_compounds = A.shape

    # For now, we stipulate that A must be elemental and therefore nonnegative
    if np.any(A < 0):
        raise ValueError("A must contain only non-negative entries")

    ret_shape = x0.shape
    if len(x0.shape) == 1:
        x0 = np.reshape(x0, (1, len(x0)), order="C")

    n_titration_points = x0.shape[0]

    # Check argument lengths
    if x0.shape[1] != n_compounds:
        raise ValueError("x0 must contain an entry for each compound.")

    x = np.empty((n_titration_points, n_compounds))

    for i_point in range(n_titration_points):
        A_new, G_new, constraint_vector, active_compounds = prune_AG(
            A, G, x0[i_point], elemental
        )

        # Return the original concentrations if there is no freedom
        if len(G_new) > 0:
            if len(A_new) == 0:
                x_new = np.exp(-G_new)
                converged = True
                run_stats = None
            else:
                x_new, converged, run_stats = eqtk_conc_optimize_pure_python(
                    A_new,
                    G_new,
                    constraint_vector,
                    trust_region_params=trust_region_params,
                    quiet=quiet,
                )

            if not converged:
                raise RuntimeError("Calculation of concentrations did not converge!")

        # Put in concentrations that were cut out
        j = 0
        for i in range(n_compounds):
            if active_compounds[i]:
                x[i_point, i] = x_new[j]
                j += 1
            else:
                x[i_point, i] = x0[i_point, i]
    x = np.reshape(x, ret_shape)

    # EDIT TO RETURN RUNSTATS
    return x, None


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
        raise ValueError("Invalid units specification.")

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
    Return value of thermal energy kT in specified units.  T is
    assumed to be in deg. C.
    """

    if T > 150.0:
        print("\nWARNING: T may be in wrong units, must be in deg. C.\n")

    T += 273.15

    if units == "kcal/mol":
        return 0.0019872041 * T
    elif units == "J":
        return 1.3806488e-23 * T
    elif units == "J/mol":
        return 8.3144621 * T
    elif units == "kJ/mol":
        return 0.0083144621 * T
    elif units == "pN-nm":
        return 0.013806488 * T
    else:
        raise ValueError("Improper thermal energy units specification.")


class TrustRegionParams(object):
    """
    Class contraining trust region parameters.
    """

    def __init__(self, **kwargs):
        """
        Pop the trust region and assign defaults
        """
        self.max_iters = kwargs.pop("max_iters", 10000)
        self.delta_bar = kwargs.pop("delta_bar", 1000.0)
        self.eta = kwargs.pop("eta", 0.125)
        self.min_delta = kwargs.pop("min_delta", 1e-12)
        self.max_trial = kwargs.pop("max_trial", 1)
        self.perturb_scale = kwargs.pop("perturb_scale", 100.0)
        self.seed = kwargs.pop("seed", 0)
        self.write_log_file = int(kwargs.pop("write_log_file", 0))
        time_str = time.strftime("%Y-%m-%d-%X", time.localtime())
        self.log_file = kwargs.pop("log_file", "eqtk_%s.log" % time_str)

        # The tolerance is a relative tolerance.  The absolute tolerance is
        # self.tol * (mininium single-species initial mole fraction)
        self.tol = kwargs.pop("tol", 0.0000001)


def calc_smooth_curve(
    N,
    K,
    x0,
    final_x,
    final_species,
    final_tolerance=1e-6,
    trust_region_params=None,
    quiet=True,
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
    trust_region_params : instance of TrustRegionParams class
        Contains pertinent parameters for trust region calculation.
        If None, defaults are used.
    quiet : Boolean, default = True
        True to suppress output to the screen.

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

        res_x, converged, res_stats = calc_conc(
            N_new, K_new, x0, trust_region_params, quiet
        )
        result_x[i, :] = res_x

    return result_x


# Use Numba'd functions
if numba_check.numba_check():
    import numba

    _thermal_energy = numba.jit(_thermal_energy, nopython=True)
    # _dimensionless_free_energy = numba.jit(_dimensionless_free_energy,
    #                                       nopython=True)
    # _water_density = numba.jit(_water_density, nopython=True)
#    _nullspace_svd = numba.jit(_nullspace_svd, nopython=True)
