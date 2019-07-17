"""
Build test cases for equilibrium solver.
"""
import itertools
import numpy as np

# pytest raises an except for a RuntimeWarning that Numpy suppresses when loading scipy.special.
# RuntimeWarning: numpy.ufunc size changed, may indicate binary incompatibility. Expected 192 from C header, got 216 from PyObject
# import scipy.special

# Below is a substitute, replace comb() below.
import math
def comb(n, k):
    return math.factorial(n) // math.factorial(k) // math.factorial(n - k)



# ###################################################################
def random_test_case(n_particles, max_compound_size, max_log_conc=-8,
                     min_log_conc=-30):
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
        mset_size[k-1] = comb(n_particles + k - 1, k)
        n_compounds += mset_size[k-1]

    # Stoichiometry matrix from multisets
    cmp_list = range(n_particles)
    A = np.empty((n_particles, n_compounds), dtype=float)
    j = 0
    for k in range(1, max_compound_size + 1):
        mset = itertools.combinations_with_replacement(cmp_list, k)
        for n in range(mset_size[k-1]):
            cmp_formula = next(mset)
            for i in range(n_particles):
                A[i,j] = cmp_formula.count(i)
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
        N[r,r+n_particles] = 1
        N[r,:n_particles] = -A[:,r+n_particles]

    # Make a new set of concentrations that have compounds
    # by successively runnings rxns to half completion
    x0 = np.concatenate((x0_particles, np.zeros(n_compounds - n_particles)))
    for r in range(n_compounds - n_particles):
        # Identify limiting reagent
        lim = 1.0
        for i in range(n_particles):
            if N[r,i] != 0:
                if x0_particles[i] / abs(N[r,i]) < lim:
                    lim_reagent = i
                    lim = x0[i]

        # Carry out reaction half way
        x0 += x0[lim_reagent] / abs(N[r,lim_reagent]) * N[r] * 0.5

    return N.astype(float), K, A, G, x0_particles, x0, x


# ###################################################################
def make_test_cases():
    """
    Generates a dictionary of test cases and a list of random test
    cases that should give proper results.  Also generates a set of
    test cases that should give errors.

    Needs to be added to and updated to include free energies.

    n_random_test_cases = number of random test cases to generate
    max_particles = maximal number of particles in random test case
    max_compound_size = maximal compound size in random test cases
    """

    # Create dictionary of erroneous test cases
    error_test_cases = {}
    # ###########
    description = 'cyclic_conversion'
    # reactions not linearly independent should give error
    N = np.array([[-1,  1,  0],
                  [ 0, -1,  1],
                  [ 1,  0, -1]])
    K = np.array([100.0, 100.0, 100.0])
    x0 = np.array([2.0, 0.05, 1.0])
    raises = ValueError
    excinfo = 'Rows in stoichiometric matrix N must be linearly independent.'
    error_test_cases[description] = TestCase(
        x0, N=N, K=K, units='M', description=description, raises=raises,
        excinfo=excinfo)



class TestCase(object):
    """
    Generates a set of test cases.
    """

    def __init__(self, x0, N=None, K=None, A=None, G=None, x=None,
                 titrated_species=0,
                 x0_titrated=None, vol_titrated=None, initial_volume=None,
                 x_titrant=None, units='M', raises=None, excinfo=None,
                 description=None):
        """
        Generate the test cases.  x0 can be a tuple of concentrations
        we wish to test.
        """
        self.x0 = x0
        self.x = x
        self.N = N
        self.K = K
        self.G = G
        self.A = A
        self.x0_titrated = x0_titrated
        self.titrated_species = titrated_species
        self.vol_titrated = vol_titrated
        self.initial_volume = initial_volume
        self.x_titrant = x_titrant
        self.units = units
        self.description = description
        self.raises = raises
        self.excinfo = excinfo

    def test_case_tuple(self):
        """
        Returns a tuple (N, K, x_0, G, A, x0_titrated, titrated_species, units, raises, excinfo,)
        from a TestCase instance.
        """
        return (self.N, self.K, self.x0, self.G, self.A, self.x0_titrated,
                self.titrated_species, self.vol_titrated, self.initial_volume,
                self.x_titrant, self.units)
