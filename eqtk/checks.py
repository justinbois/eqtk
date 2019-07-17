import numpy as np
import eqtk
from . import linalg

def check_equilibrium_NK(c0, c, N=None, K=None):
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

    Returns
    -------
    equilibrium_err : array_like, shape(n_compounds - n_particles,)
        equilibrium_err[r] is the fractional error in the
        equilibrium expression for chemical reaction r.  This is
        defined as 1 - K[r] / prod_j c[j]**N[r][j]
    cons_mass_err : array_like, shape(n_particles,)
        cons_mass_err[i] is the error in conservation of mass for
        irreducible species (particle) i.  It is defined as
        cons_mass_err = (c0 - np.dot(A, c)) / c0

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
    >>> equilibrium_err, cons_mass_err = eqtk.check_equilibrium(c, A, N, K, c0)
    >>> equilibrium_err
    array([  3.33066907e-16,  -4.66293670e-15,  -4.44089210e-16,
             0.00000000e+00])
    >>> cons_mass_err
    array([ -4.11757295e-11,  -1.28775509e-11])
    """
    c0_corr, N_corr, K_corr, _, _ = check_eq_input(c0, N, K)

    N_new, minus_log_K_new, c0_new, active_compounds, active_reactions = eqtk.prune_NK(
        N_corr, -np.log(K_corr), c0_corr
    )

    # Check equilibrium expressions
    eq_ok = np.empty(len(active_reactions))
    r_active = 0
    for r, K in enumerate(K_corr):
        if active_reactions[r]:
            rhs = np.dot(N_new[r_active], np.log(c[active_compounds]))
            eq_ok[r] = np.isclose(np.exp(rhs - np.log(K)), 1)
            r_active += 1
        else:
            eq_ok[r] = 1

    # Check conservation expressions
    A = linalg.nullspace_svd(N_corr).transpose()
    target = np.dot(A, c0_corr)
    res = np.dot(A, c)
    cons_mass_ok = np.empty(len(res))
    for i, (res_val, target_val) in enumerate(zip(res, target)):
        if target_val == 0:
            if res_val != 0:
                cons_mass_ok[i] = False
            else:
                cons_mass_ok[i] = True
        else:
            cons_mass_ok[i] = np.isclose(res_val, target_val)

    return eq_ok, cons_mass_ok


def check_equilibrium_AG(
    c0, c, A=None, G=None, T=None, solvent_density=None, units=None, G_units=None
):
    """Check equilibrium for the AG formulation.

    Equilibrium conditions are satisfied by construction; we need only
    to check conservation of mass.
    """
    return 1 - np.dot(A, c) / np.dot(A, c0)


def check_eq_input(c0, N=None, K=None, A=None, G=None):
    """
    Utility to check input to the function eqtk_conc.
    Does appropriate transposing and typing and returns result.

    Parameters
    ----------
    c0 : array_like, shape (n_titration_points, n_compounds)
        Initial concentration of all species in solution.
    N : array_like, shape (n_compounds - n_particles, n_compounds)
        N[r][j] = the stoichiometric coefficient of compounds j
        in chemical reaction r.
    K : array_like, shape (n_compounds - n_particles,)
        K[r] is the equilibrium constant for chemical reaction r
    A : array_like, shape (n_constraints, n_compounds)
        A[i][j] = number of particles of type i in compound j.
    G : array_like, shape (n_compounds,)
        G[j] is the free energy of compound j in units of kT.

    Returns
    -------
    corr_c0 : array_like, shape (n_compounds,)
        A corrected version of c0 suitable for input to
        calculate_concentrations.
    corr_N : array_like, shape (n_compounds - n_particles, n_compounds)
        A corrected version of N suitable for input to
        calculate_concentrations.
    corr_K : array_like, shape (n_compounds - n_particles,)
        A corrected version of K suitable for input to
        calculate_concentrations.
    corr_A : array_like, shape (n_particles, n_compounds)
        A corrected version of A suitable for input.
    corr_G : array_like, shape (n_compounds,)
        A corrected version of G suitable for input.

    Raises
    ------
    ValueError
      If input is invalid and not fixable for input to
      calculate_concentrations.

    Examples
    --------
    1) Inputted N is ints, K given as tuple of ints. These are
       correct on return.

    >>> import numpy as np
    >>> import eqtk
    >>> N = np.array([[-1,  0,  1,  0,  0,  0],
                      [-1, -1,  0,  1,  0,  0],
                      [ 0, -2,  0,  0,  1,  0],
                      [ 0, -1, -1,  0,  0,  1]])
    >>> K = (50, 10, 40, 100)
    >>> c0 = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    >>> corr_c0, corr_N, corr_K, _, _ = eqtk.check_eq_input(c0, N, K,
                                                        A=None, G=None)
    >>> corr_N
    array([[-1.,  0.,  1.,  0.,  0.,  0.],
           [-1., -1.,  0.,  1.,  0.,  0.],
           [ 0., -2.,  0.,  0.,  1.,  0.],
           [ 0., -1., -1.,  0.,  0.,  1.]])
    >>> corr_K
    array([  50.,   10.,   40.,  100.])
    """

    # Make sure inputs are ok
    if (N is None or K is None) and (A is None or G is None):
        raise ValueError("Must specify either N/K pair of A/G pair.")

    # Check c0 first (it's required)
    if type(c0) == list or type(c0) == tuple or not c0.flags["C_CONTIGUOUS"]:
        c0 = np.array(c0, order="C")

    c0 = c0.astype(float)

    if len(c0.shape) == 1:
        n_compounds = c0.shape[0]
    elif len(np.shape(c0)) == 2:
        n_compounds = c0.shape[1]
    else:
        raise ValueError("c0 is the wrong shape.")

    # Initialize in case N isn't defined
    n_reactions = 0

    # Check N
    if N is not None:
        if type(N) == list or type(N) == tuple:
            N = np.array(N)

        N = N.astype(float)

        if len(N.shape) == 1:
            N = N.reshape((1, -1), order="C").astype(float)

        if not N.flags["C_CONTIGUOUS"]:
            N = np.array(N, order="C").astype(float)

        n_reactions = N.shape[0]

    # Check K
    if K is not None:
        if type(K) == list or type(K) == tuple:
            K = np.array(K)
        K = K.astype(float)

        if len(np.shape(K)) == 2 and (np.shape(K)[1] == 1 or np.shape(K)[0] == 1):
            K = K.flatten()
        elif len(np.shape(K)) != 1:
            raise ValueError("K is the wrong shape.")

    # Check A
    if A is not None:
        if type(A) == list or type(A) == tuple:
            A = np.array(A, order="C").astype(float)

        if not A.flags["C_CONTIGUOUS"]:
            A = np.array(A, order="C").astype(float)
        A = A.astype(float)

    # Check G
    if G is not None:
        if type(G) == list or type(G) == tuple:
            G = np.array(G, order="C")
        G = G.astype(float)
        if len(np.shape(G)) == 2 and (np.shape(G)[1] == 1 or np.shape(G)[0] == 1):
            G = G.flatten()
        elif len(np.shape(G)) != 1:
            raise ValueError("G is the wrong shape.")

    # If both N and A are given, check to make sure rows of A span null(N)
    if A is not None and N is not None:
        if abs(np.dot(N, A.transpose())).max() > 1e-14:
            raise ValueError("Rows of A do not span null space of N.")

    # If N, K, and G are all given, check consistency
    if N is not None and K is not None and G is not None:
        if abs(np.dot(N, G) + np.log(K)).max() > 1e-14:
            raise ValueError("K and G are inconsistent.")

    # Check for consistency
    if N is not None and N.shape[1] != n_compounds:
        raise ValueError("c0 and N must have same number of columns.")

    if A is not None and A.shape[1] != n_compounds:
        raise ValueError("c0 and A must have same number of columns.")

    if K is not None:
        if len(K) != n_reactions:
            raise ValueError("Wrong number of equilibrium constants!")
        if not (K > 0.0).all():
            raise ValueError("All K's must be positive!")
        if np.isinf(K).any():
            raise ValueError("All K's must be finite!")

    if G is not None:
        if len(G) != n_compounds:
            raise ValueError("Wrong number of free energies!")
        if np.isinf(G).any():
            raise ValueError("All G's must be finite!")

    if np.isnan(c0).any():
        raise ValueError("c0 has NaN!")
    if (c0 < 0.0).any():
        raise ValueError("All c0's must be nonnegative!")
    if np.isinf(c0).any():
        raise ValueError("All c0's must be finite!")

    return c0, N, K, A, G
