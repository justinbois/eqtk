import numpy as np
import pandas as pd

from . import eqtk
from . import linalg
from . import parsers
from . import constants


def _c_from_df(c):
    """Extract concentrations from outputted data frame"""
    if type(c) == pd.core.series.Series:
        return c[c.index[~c.index.str.contains("]__0")]]
    elif type(c) == pd.core.frame.DataFrame:
        return c[c.columns[~c.columns.str.contains("]__0")]]

    return c


def check_equilibrium_NK(c0, c, N=None, K=None):
    """Check to make sure equilibrium is satisfied."""
    c0, N, K, _, _, names, _, single_point = parsers.parse_input(
        c0, N, K, *tuple([None] * 7)
    )

    c = _c_from_df(c)
    c, _, _, _, _, _, _, _ = parsers.parse_input(c, N, K, *tuple([None] * 7))

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
    A = linalg.nullspace_svd(N, constants.nullspace_tol)
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

    c0, _, _, A, G, _, _, _ = parsers.parse_input(
        c0, None, None, A, G, *tuple([None] * 5)
    )

    c = _c_from_df(c)
    c, _, _, _, _, _, _, _ = parsers.parse_input(
        c, None, None, A, G, *tuple([None] * 5)
    )

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
