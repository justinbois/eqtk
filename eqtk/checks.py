import numpy as np
from . import eqtk
from . import linalg


def check_equilibrium_NK(c0, c, N=None, K=None):
    """Check to make sure equilibrium is satisfied."""
    single_point = False
    if len(c0.shape) == 1:
        single_point = True

    c0, N, K, _, _ = check_input(c0, N, K, *tuple([None]*6))
    c, _, _, _, _ = check_input(c, N, K, *tuple([None]*6))
       
    if c0.shape != c.shape:
        raise ValueError("`c0` and `c` must have the same shape.")

    eq_ok = np.empty((c.shape[0], N.shape[0]), dtype=bool)
    cons_mass_ok = np.empty((c.shape[0], N.shape[1] - N.shape[0]), dtype=bool)
    for i, (c0_i, c_i) in enumerate(zip(c0, c)):
        eq_ok_i, cons_mass_ok_i = _check_equilibrium_NK_single_point(c0_i, c_i, N, K)
        eq_ok[i] = eq_ok_i
        cons_mass_ok[i] = cons_mass_ok_i

    if single_point:
        return eq_ok.flatten(), cons_mass_ok.flatten()

    return eq_ok, cons_mass_ok


def _check_equilibrium_NK_single_point(c0, c, N=None, K=None):
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
    _, _, _, _, active_reactions = eqtk._prune_NK(N, -np.log(K), c0)

    # Check equilibrium expressions
    eq_ok = np.empty(len(active_reactions), dtype=bool)
    logc = np.array([np.log(ci) if ci > 0 else 0.0 for ci in c])
    for r, K in enumerate(K):
        if active_reactions[r]:
            rhs = np.dot(N[r], logc)
            eq_ok[r] = np.isclose(np.exp(rhs - np.log(K)), 1)
        else:
            eq_ok[r] = 1

    # Check conservation expressions
    A = linalg.nullspace_svd(N).transpose()
    c0_adjusted = eqtk._create_from_nothing(N, c0)
    target = np.dot(A, c0_adjusted)
    res = np.dot(A, c)
    cons_mass_ok = np.empty(len(res), dtype=bool)
    for i, (res_val, target_val) in enumerate(zip(res, target)):
        if target_val == 0:
            if res_val != 0:
                cons_mass_ok[i] = False
            else:
                cons_mass_ok[i] = True
        else:
            cons_mass_ok[i] = np.isclose(res_val, target_val)

    return eq_ok, cons_mass_ok


def check_equilibrium_AG(c0, c, A, G):
    """Check to make sure equilibrium is satisfied."""
    single_point = False
    if len(c0.shape) == 1:
        single_point = True

    c0, _, _, A, G = check_input(c0, None, None, A, G, *tuple([None]*4))
    c, _, _, _, _ = check_input(c, None, None, A, G, *tuple([None]*4))

    if c0.shape != c.shape:
        raise ValueError("`c0` and `c` must have the same shape.")

    cons_mass_ok = np.empty((c.shape[0], A.shape[0]), dtype=bool)
    for i, (c0_i, c_i) in enumerate(zip(c0, c)):
        cons_mass_ok_i = _check_equilibrium_AG_single_point(c0_i, c_i, A, G)
        cons_mass_ok[i] = cons_mass_ok_i

    if single_point:
        return cons_mass_ok.flatten()

    return cons_mass_ok


def _check_equilibrium_AG_single_point(c0, c, A, G):
    """Check equilibrium for the AG formulation.

    Equilibrium conditions are satisfied by construction; we need only
    to check conservation of mass.
    """
    target = np.dot(A, c0)
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

    return cons_mass_ok


def check_input(c0, N, K, A, G, units, solvent_density, T, G_units):
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
    >>> corr_c0, corr_N, corr_K, _, _ = eqtk.check_input(c0, N, K,
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
    err_str = "Must specify either N/K pair or A/G pair."
    if N is None:
        if A is None or G is None:
            raise RuntimeError(err_str)
    elif A is not None or G is not None:
        raise RuntimeError(err_str)

    # Check units
    allowed_units = (None, "M", "mM", "uM", "µM", "nM", "pM")
    if units not in allowed_units:
        raise ValueError(
            f"Specified concentration units of {units} not in {allowed_units}."
        )

    allowed_units = (None, "kT", "kcal/mol", "J", "J/mol", "kJ/mol", "pN-nm")
    if G_units not in allowed_units:
        raise ValueError(
            f"Specified free energy units of {G_units} not in {allowed_units}."
        )

    if G_units not in (None, "kT") and T is None:
        raise ValueError("If G is specified with units, must also supply T.")

    # Make sure T in in Kelvin.
    if T is not None and T < 100.0:
        warnings.warn("WARNING: T may be in wrong units, must be in Kelvin.")

    # Make sure solvent density is ok
    if solvent_density is not None and units is None and solvent_density != 1.0:
        raise ValueError(
            "If `solvent_density` is specified, `units` must also be specified."
        )

    # Check c0
    if type(c0) == list or type(c0) == tuple or not c0.flags["C_CONTIGUOUS"]:
        c0 = np.array(c0, order="C")

    c0 = c0.astype(float)

    if len(c0.shape) == 1:
        n_compounds = c0.shape[0]
        c0 = np.expand_dims(c0, axis=0)
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

        if type(N) == np.ndarray:
            if K is None:
                raise ValueError('`K` must be specified.')
            N = N.astype(float)

            if len(N.shape) == 1:
                N = N.reshape((1, -1), order="C").astype(float)

            N = np.ascontiguousarray(N, dtype=float)
        elif type(N) == pandas.core.frame.DataFrame:
            N, names, K, units_from_N = _NK_and_names_from_df(N)
            if K is None:
                raise ValueError('`K` must be specified either separately or in `N` data frame.')
            K = _convert_K(K_units, units)

        if K is None:
            raise ValueError("`K` not specified.")

        if np.isinf(N).any():
            raise ValueError("All entries in N must be finite.")
        if np.isnan(N).any():
            raise ValueError("No NaN values are allowed in N.")

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

        if len(A.shape) == 1:
            A = A.reshape((1, -1), order="C").astype(float)

        # Make sure empty constraint matrix has correct dimensions
        if A is not None and np.sum(A.shape) == 1:
            A = A.reshape((0, 1))

        if np.isinf(A).any():
            raise ValueError("All entries in A must be finite.")
        if np.isnan(A).any():
            raise ValueError("No NaN values are allowed in A.")

        if np.any(A < 0):
            raise ValueError("A must have all nonnegative entries.")

        A = np.ascontiguousarray(A, dtype=float)

    # Check G
    if G is not None:
        if type(G) == list or type(G) == tuple:
            G = np.array(G, order="C")
        G = G.astype(float)
        if len(np.shape(G)) == 2 and (np.shape(G)[1] == 1 or np.shape(G)[0] == 1):
            G = G.flatten()
        elif len(np.shape(G)) != 1:
            raise ValueError("G is the wrong shape.")

    # Check for consistency
    if N is not None and N.shape[1] != n_compounds:
        raise ValueError("c0 and N must have same number of columns.")

    if A is not None and A.shape[1] != n_compounds:
        raise ValueError("c0 and A must have same number of columns.")

    if K is not None:
        if len(K) != n_reactions:
            raise ValueError("K must have N.shape[0] entries")
        if not (K > 0.0).all():
            raise ValueError("All K's must be positive.")
        if np.isinf(K).any():
            raise ValueError("All K's must be finite.")
        if np.isnan(K).any():
            raise ValueError("No NaN values are allowed for K.")

    if G is not None:
        if len(G) != n_compounds:
            raise ValueError("G must have A.shape[1] entries.")
        if np.isinf(G).any():
            raise ValueError("All G's must be finite.")
        if np.isnan(G).any():
            raise ValueError("No NaN values are allowed for G.")

    if np.isnan(c0).any():
        raise ValueError("No NaN values are allowed for c0.")
    if (c0 < 0.0).any():
        raise ValueError("All c0's must be nonnegative.")
    if np.isinf(c0).any():
        raise ValueError("All c0's must be finite!")

    # Ensure N and A have full row rank
    if N is not None and len(N) > 0 and np.linalg.matrix_rank(N) != n_reactions:
        raise ValueError("N must have full row rank.")
    if A is not None and len(A) > 0 and np.linalg.matrix_rank(A) != A.shape[0]:
        raise ValueError("A must have full row rank.")

    return c0, N, K, A, G


def _NK_and_names_from_df(df):
    cols = []
    K = None
    units = None
    for col in df.columns:
        if 'equilibrium constant' or 'equilibrium_constant' in col:
            K = df[col].values.astype(float)
        else:
            cols.append(col)

    N = df[cols].to_numpy(dtype=float, copy=True)
    names = list(cols)

    for name in names:
        if _levenshtein(name, 'equilibrium constant') < 10:
            warnings.warn(f"Chemical species name '{name}' is close to 'equilibrium constant'. This may be a typo.")

    return N, K, names


def _A_and_names_from_df(df, solvent_density):
    A = df[cols].to_numpy(dtype=float, copy=True)
    names = list(cols)

    return A, names


def _c0_from_df(df, names):
    c0 = np.empty((len(df), len(names)))
    for i, name in enumerate(names):
        c0[i,:] = df[name].values

    return c0.astype(float)


def _levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]
