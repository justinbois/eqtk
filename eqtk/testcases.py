import itertools
import math

import numpy as np
import pandas as pd

import eqtk


def comb(n, k):
    """We use this custom function to avoid pytest warnings for Scipy
    built against wrong libraries that sometimes comes up."""
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)


def random_elemental_test_case(
    n_particles, max_compound_size, max_log_conc=-8, min_log_conc=-30
):
    """
    Generates A, N, K, and G for a set of compounds made of
    n_particles different types of particles.  All units are
    dimensionless. Particles have energy zero by definition.

    The number of compounds with different stoichiometry consisting
    of k total particles that can be generated from n particle types
    is (n + k -1) choose k.
    """
    n_compounds = 0
    mset_size = np.empty(max_compound_size, dtype=int)
    for k in range(1, max_compound_size + 1):
        #        mset_size[k-1] = scipy.special.comb(n_particles + k - 1, k, exact=True)
        mset_size[k - 1] = comb(n_particles + k - 1, k)
        n_compounds += mset_size[k - 1]

    # Elemental conservation matrix from multisets
    cmp_list = range(n_particles)
    A = np.empty((n_particles, n_compounds), dtype=float)
    j = 0
    for k in range(1, max_compound_size + 1):
        mset = itertools.combinations_with_replacement(cmp_list, k)
        for n in range(mset_size[k - 1]):
            cmp_formula = next(mset)
            for i in range(n_particles):
                A[i, j] = cmp_formula.count(i)
            j += 1

    # Generate random concentrations
    log_x = np.random.rand(n_compounds) * (max_log_conc - min_log_conc) + min_log_conc
    x = np.exp(log_x)

    # Generate G's from conc's (particles have energy zero by definition)
    G = -log_x + np.dot(A.transpose(), log_x[:n_particles])
    G[:n_particles] = 0.0

    # Generate K's from G's
    K = np.exp(-G[n_particles:])

    # Get total concentration of particle species
    x0_particles = np.dot(A, x)

    # Generate N from A
    N = np.zeros((n_compounds - n_particles, n_compounds), dtype=int)
    for r in range(n_compounds - n_particles):
        N[r, r + n_particles] = 1
        N[r, :n_particles] = -A[:, r + n_particles]

    # Make a new set of concentrations that have compounds
    # by successively runnings rxns to half completion
    x0 = np.concatenate((x0_particles, np.zeros(n_compounds - n_particles)))
    for r in range(n_compounds - n_particles):
        # Identify limiting reagent
        lim_reagent_array = [
            x0[i] / np.abs(N[r, i]) if N[r, i] != 0 else np.inf
            for i in range(n_particles)
        ]
        lim_reagent = np.argmin(lim_reagent_array)

        # Carry out reaction half way
        x0 += x0[lim_reagent] / abs(N[r, lim_reagent]) * N[r] * 0.5

    # Generate names
    names = ["{0:08d}".format(i) for i in range(n_compounds)]
    N_df = pd.DataFrame(data=N, columns=names)
    N_df["equilibrium constant"] = K
    x_series = eqtk.to_df(x, x0, names=names)
    x_series_log = eqtk.to_df(log_x, x0, names=names)

    return dict(
        c0=x0,
        N=N,
        K=K,
        A=A,
        G=G,
        c=x,
        c_series=x_series,
        c_log=log_x,
        c_series_log=x_series_log,
        N_df=N_df,
        units=None,
    )


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
    c_log = np.log(c)
    K = np.array([Kd], dtype=float)
    N = np.array([[1, 1, -1]], dtype=float)
    A = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    G = np.array([0, 0, np.log(Kd)], dtype=float)

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

    c_series_log = pd.Series(
        {
            "[A]__0": cA0,
            "[B]__0": cB0,
            "[AB]__0": 0.0,
            "ln [A]": np.log(c[0]),
            "ln [B]": np.log(c[1]),
            "ln [AB]": np.log(c[2]),
        }
    )

    N_df = pd.DataFrame(data=N, columns=["A", "B", "AB"])
    N_df["equilibrium constant"] = K

    return dict(
        c0=c0,
        N=N,
        K=K,
        A=A,
        G=G,
        c=c,
        c_series=c_series,
        c_log=c_log,
        c_series_log=c_series_log,
        N_df=N_df,
        units=None,
    )
