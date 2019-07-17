import random
import numpy as np
import eqtk
import eqtk_test_cases


def test_prune_NK():
    N = np.array(
        [
            [-1, 1, 0, 0, 0, 0],
            [-1, 0, -1, 1, 0, 0],
            [0, -2, 0, 0, 1, 0],
            [0, -1, -1, 0, 0, 1],
        ]
    )
    minus_log_K = np.array([1, 2, 3, 4])

    # All present
    x0 = np.array([1, 2, 3, 4, 5, 6])
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Also all present
    x0 = np.array([1, 0, 3, 0, 0, 0])
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Also all present
    x0 = np.array([0, 0, 0, 0, 0, 6])
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Only entries 0, 1 and 4 present
    x0 = np.array([0, 2, 0, 0, 0, 0])
    x0_target = np.array([0, 2, 0])
    N_target = np.array([[-1, 1, 0], [0, -2, 1]])
    minus_log_K_target = np.array([1, 3])
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N_target)
    assert np.array_equal(minus_log_K_new, minus_log_K_target)
    assert np.array_equal(x0_new, x0_target)

    # Only entry 2 present
    x0 = np.array([0, 0, 3, 0, 0, 0])
    N_target = np.array([[]])
    minus_log_K_target = np.array([])
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N_target)
    assert np.array_equal(minus_log_K_new, minus_log_K_target)
    assert np.array_equal(x0_new, x0)

    N = np.array(
        [
            [-1, 0, 1, 0, 0, 0],
            [-1, -1, 0, 1, 0, 0],
            [0, -2, 0, 0, 1, 0],
            [0, -1, -1, 0, 0, 1],
        ]
    )
    minus_log_K = np.array([1, 2, 3, 4])

    # All present
    x0 = np.array([1, 2, 3, 4, 5, 6])
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Also all present
    x0 = np.array([1, 2, 0, 0, 0, 0])
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Also all present
    x0 = np.array([0, 0, 0, 0, 0, 6])
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Only entries 0 and 2 present
    x0 = np.array([1, 0, 3, 0, 0, 0])
    x0_target = np.array([1, 3])
    N_target = np.array([[-1, 1]])
    minus_log_K_target = np.array([1])
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N_target)
    assert np.array_equal(minus_log_K_new, minus_log_K_target)
    assert np.array_equal(x0_new, x0_target)

    # Only entries 1 and 4 present
    x0 = np.array([0, 2, 0, 0, 0, 0])
    x0_target = np.array([2, 0])
    N_target = np.array([[-2, 1]])
    minus_log_K_target = np.array([3])
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N_target)
    assert np.array_equal(minus_log_K_new, minus_log_K_target)
    assert np.array_equal(x0_new, x0_target)


def test_prune_AG():
    A = np.array([[1, 0, 1, 1, 0, 1], [0, 1, 0, 1, 2, 1]]).astype(float)
    G = np.array([1, 2, 3, 4, 5, 6])

    # No pruning
    x0 = np.array([1, 2, 3, 4, 5, 6])
    A_new, G_new, x0_new, _ = eqtk.prune_AG(A, G, x0, True)
    assert np.array_equal(A_new, A)
    assert np.array_equal(G_new, G)
    assert np.array_equal(x0_new, np.dot(A, x0))

    # No pruning
    x0 = np.array([1, 2, 0, 0, 0, 0])
    A_new, G_new, x0_new, _ = eqtk.prune_AG(A, G, x0, True)
    assert np.array_equal(A_new, A)
    assert np.array_equal(G_new, G)
    assert np.array_equal(x0_new, np.dot(A, x0))

    # No pruning
    x0 = np.array([0, 0, 0, 0, 0, 6])
    A_new, G_new, x0_new, _ = eqtk.prune_AG(A, G, x0, True)
    assert np.array_equal(A_new, A)
    assert np.array_equal(G_new, G)
    assert np.array_equal(x0_new, np.dot(A, x0))

    # Only species 1 and 4
    x0 = np.array([0, 0, 0, 0, 5, 0])
    x0_prune = np.array([0, 5])
    A_target = np.array([[1, 2]])
    G_target = np.array([2, 5])
    A_new, G_new, x0_new, _ = eqtk.prune_AG(A, G, x0, True)
    assert np.array_equal(A_new, A_target)
    assert np.array_equal(G_new, G_target)
    assert np.array_equal(x0_new, np.dot(A_new, x0_prune))


def test_trivial():
    N = np.array([[1]]).astype(float)
    K = np.array([0.5]).astype(float)
    A = np.array([[]]).reshape((0, 1)).astype(float)
    G = -np.log(K)
    x0 = np.array([0]).astype(float)
    res_NK, st_NK = eqtk.eqtk_conc_pure_python(N, -np.log(K), x0)
    res_AG, st_AG = eqtk.eqtk_conc_from_free_energies_pure_python(A, G, x0)
    assert np.allclose(res_NK, K)
    assert np.allclose(res_AG, K)


def test_random_cases(n_random_test_cases=100, max_particles=4, max_compound_size=5):
    # Generate list of random test cases
    random_test_cases = []
    for i in range(n_random_test_cases):
        desc = "random_" + str(i)
        n_parts = np.random.randint(1, max_particles)
        max_cmp_size = np.random.randint(1, max_compound_size + 1)
        N, K, A, G, x0_particles, x0, x = eqtk_test_cases.random_test_case(
            n_parts, max_cmp_size
        )
        random_test_cases.append(
            eqtk_test_cases.TestCase(
                x0, N=N, K=K, A=A, G=G, units=None, description=desc, x=x
            )
        )

    for tc in random_test_cases:
        res_NK, st_NK = eqtk.eqtk_conc_pure_python(tc.N, -np.log(tc.K), tc.x0)
        res_AG, st_AG = eqtk.eqtk_conc_from_free_energies_pure_python(tc.A, tc.G, tc.x0)
        # assert rs_NK.n_chol_fail_cauchy_steps == 0
        # assert rs_NK.n_irrel_chol_fail == 0
        # assert rs_NK.n_dogleg_fail == 0
        # assert rs_AG.n_chol_fail_cauchy_steps == 0
        # assert rs_AG.n_irrel_chol_fail == 0
        # assert rs_AG.n_dogleg_fail == 0
        assert np.allclose(res_NK, tc.x)
        assert np.allclose(res_AG, tc.x)


def test_scale_factor_failure():
    A = np.array([[1.0, 0.0, 2.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 2.0]])
    G = np.array([0.0, 0.0, 0.77428976, -5.64873697, -0.95863043])
    x0 = np.array(
        [
            [
                5.50293892e-05,
                6.49273515e-08,
                2.75796219e-05,
                1.29854703e-07,
                3.24636758e-08,
            ]
        ]
    )
    x = eqtk.eqtk_conc_from_free_energies_pure_python(A, G, x0)


def test_past_failures():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    G = np.array([0.0, 0.0])
    x0 = np.array([[3.48219906e-06, 1.32719868e-10]])
    elemental = False

    A = np.array([[1.0, 0.0, 2.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0, 2.0]])
    G = np.array([0.0, 0.0, -16.76857677, -2.38430181, 1.22028775])
    x0 = np.array(
        [
            [
                1.65989040e-10,
                1.07630096e-04,
                1.65989040e-10,
                1.65989040e-10,
                5.38150479e-05,
            ]
        ]
    )
    elemental = False


    N = np.array([[-2.,  1.,  0.,  0.],
                  [-3.,  0.,  1.,  0.],
                  [-4.,  0.,  0.,  1.]])
    minus_log_K = np.array([-43.66660344, -68.14676841, -92.28023823])
    x0 = np.array([[1.87852623e-06, 3.75705246e-06, 1.25235082e-06, 4.69631557e-07]])


    A = np.array([[1., 0.],
                  [0., 1.]])
    G = np.array([0., 0.])
    x0 = np.array([[2.24222410e-08, 1.63359284e-04]])
    elemental = False


    A = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]])
    G = np.array([0., 0., 0.])
    x0 = np.array([[2.63761955e-04, 4.93360042e-07, 4.88340687e-07]])
    elemental = False


# # Tests
# class NKTests(ut.TestCase):

#     def assertEqualTolerance(self, a, b, sig_figs):
#         aa = abs(a)
#         ab = abs(b)
#         if aa == 0 or ab == 0:
#             self.assertTrue(aa < 10**(-sig_figs) and ab < 10**(-sig_figs))
#         else:
#             relative_diff = 2 * abs(a - b) / (aa + ab)
#             if relative_diff > 0:
#                 dec_place = - np.log(relative_diff) / np.log(10)
#             else:
#                 dec_place = sig_figs + 1.
#             self.assertTrue(
#                 dec_place >= sig_figs,msg \
#                     = "%.10le != %.10le agree to %lf of %i decimal places" \
#                     % (a,b,dec_place, sig_figs))

#     def check_equilibrium(self, N, K, x):
#         n_reacs = N.shape[0]
#         n_comps = N.shape[1]
#         if len(x.shape) == 1:
#             x = x.reshape((1, len(x)))
#         for xpt in x:
#             for k in range(n_reacs):
#                 eqval = 1
#                 for j in range(n_comps):
#                     if xpt[j] > 0:
#                         eqval *= xpt[j]**N[k,j]
#                     else:
#                         eqval = 0
#                 if eqval > 0:
#                     self.assertEqualTolerance(eqval, K[k], TOL_DEC_PLACE)

#     def check_mass_conservation(self, A, x, x0):
#         xinit = np.dot(A,x0)
#         xfinal = np.dot(A,x)
#         for xii,xfi in zip(xinit,xfinal):
#             self.assertEqualTolerance(xii, xfi, TOL_DEC_PLACE)

#     def test_basic(self):
#         # Use default trust_region params, with one one trial
#         trust_region_params = equ.TrustRegionParams(max_trial=1)

#         N = np.array([[-1,-1,1]])
#         K = np.array([2e3])
#         x0 = np.array([3e-3,5e-3,0.0])
#         x, converged, run_stats = equ.eqtk_conc(
#             N, -np.log(K), x0, trust_region_params=trust_region_params)
#         x2, run_stats = equ.eqtk_conc_pure_python(
#             N, -np.log(K), x0, trust_region_params=trust_region_params)

#         self.assertTrue(converged)

#         for xi,xi2 in zip(x,x2):
#             self.assertEqualTolerance(xi,xi2,TOL_DEC_PLACE)
#         self.assertEqualTolerance(x[2] / (x[0]*x[1]), K, TOL_DEC_PLACE)

#         N = np.array([[-2,0,1,0,0],[-1,-1,0,1,0],[0,-2,0,0,1]])
#         K = np.array([2e4,3e4,8e3])
#         x0_1 = np.array([  3e-4,  5e-5,     0,     0,     0])
#         x0_2 = np.array([     0,     0,1.5e-4,     0,2.5e-5])
#         x0_3 = np.array([  7e-5,  1e-5,1.0e-4,  3e-5,  5e-6])
#         A = np.array([[1,0,2,1,0],[0,1,0,1,2]])


#         # Write log file for first one
#         trust_region_params.write_log_file = True
#         x_1, converged, run_stats = equ.eqtk_conc(
#             N, -np.log(K), x0_1, trust_region_params=trust_region_params)
#         trust_region_params.write_log_file = False
#         x_2, converged, run_stats = equ.eqtk_conc(
#             N, -np.log(K), x0_2, trust_region_params=trust_region_params)
#         x_3, converged, run_stats = equ.eqtk_conc(
#             N, -np.log(K), x0_3, trust_region_params=trust_region_params)
#         x_4, run_stats = equ.eqtk_conc_pure_python(
#             N, -np.log(K), x0_1, trust_region_params=trust_region_params)
#         x_5, run_stats = equ.eqtk_conc_pure_python(
#             N, -np.log(K), x0_2, trust_region_params=trust_region_params)
#         x_6, run_stats = equ.eqtk_conc_pure_python(
#             N, -np.log(K), x0_3, trust_region_params=trust_region_params)

#         for xi1,xi2,xi3,xi4,xi5,xi6 in zip(x_1,x_2,x_3,x_4,x_5,x_6):
#             self.assertEqualTolerance(xi1,xi2,TOL_DEC_PLACE)
#             self.assertEqualTolerance(xi1,xi3,TOL_DEC_PLACE)
#             self.assertEqualTolerance(xi1,xi4,TOL_DEC_PLACE)
#             self.assertEqualTolerance(xi1,xi5,TOL_DEC_PLACE)
#             self.assertEqualTolerance(xi1,xi6,TOL_DEC_PLACE)

#         self.check_equilibrium(N, K, x_1)

#         A = np.array([[1,0,2,1,0],[0,1,0,1,2]])
#         self.check_mass_conservation(A, x_1, x0_1)

#     def test_linear_dep_exceptions(self):
#         N_dep = np.array([[-1, -1, 1, 1],[-2,0,2,0],[0,-2,0,2]])
#         K_dep = np.array([1,1,1])
#         x0 = np.array([1e-5,1e-5,1e-5,1e-5])
#         self.assertRaises(ValueError, equ.eqtk_conc, N_dep, K_dep, x0)

#     def test_shape_exceptions(self):
#         N = np.array([[-1,-1,1,1]])
#         K = np.array([1,1])
#         x0 = np.array([1e-5,1e-5,1e-5,1e-5])
#         self.assertRaises(ValueError, equ.eqtk_conc, N, K, x0)

#         K = np.array([1])
#         x0 = np.array([1e-5])
#         self.assertRaises(ValueError, equ.eqtk_conc, N, K, x0)

#     def test_multiset(self):
#         for i in range(25):
#             # Use default trust_region params, with one one trial
#             trust_region_params = equ.TrustRegionParams(max_trial=1)

#             n_parts = random.randint(1,4)
#             max_compound_size = random.randint(1,5)
#             A, N, K, G, x0, x0_full, x = eqtest.random_test_case(
#                 n_parts, max_compound_size)

#             xcalc1, converged, runstats = equ.eqtk_conc(
#                 N, -np.log(K), x0_full, trust_region_params=trust_region_params)
#             self.assertTrue(converged)

#             # Check mass conservation
#             xf = np.dot(A, xcalc1.T)
#             tol_dec = int(-np.log(max(x0))/np.log(10) + TOL_DEC_PLACE-1)

#             for x0i, xfi in zip(x0, xf):
#                 self.assertAlmostEqual(x0i, xfi, tol_dec)

#             # Check equilibrium constants
#             self.check_equilibrium(N, K, xcalc1)

#             # Check against pure_python implementation
#             xcalc2, runstats = equ.eqtk_conc_pure_python(
#                 N, -np.log(K), x0_full, trust_region_params=trust_region_params)

#             for xi1, xi2 in zip(xcalc1, xcalc2):
#                 self.assertAlmostEqual(xi1, xi2, tol_dec)

#             # Reverse the direction
#             K = 1 / K
#             N = -N

#             xcalc3, converged, runstats = equ.eqtk_conc(
#                 N, -np.log(K), x0_full, trust_region_params=trust_region_params)
#             self.assertTrue(converged)

#             for xi1, xi2 in zip(xcalc1,xcalc2):
#                 self.assertAlmostEqual(xi1, xi2, tol_dec)

#             xcalc4, runstats = equ.eqtk_conc_pure_python(
#                 N, -np.log(K), x0_full, trust_region_params=trust_region_params)
#             self.assertTrue(converged)

#             for xi1, xi2 in zip(xcalc1, xcalc4):
#                 self.assertAlmostEqual(xi1, xi2, tol_dec)

#     def test_basic_eqtk(self):
#         N = np.array([[-1,-1,1]])
#         K = np.array([2])
#         x0 = np.array([3,5,0])
#         x = eq.sweep_titration(N, K, x0, units='mM')

#         self.assertEqualTolerance(x[2] / (x[0]*x[1]), K[0], TOL_DEC_PLACE)

#         N = np.array([[-2,0,1,0,0],[-1,-1,0,1,0],[0,-2,0,0,1]])
#         K = np.array([2e4,3e4,8e3])
#         x0_1 = np.array([  3e-4,  5e-5,     0,     0,     0])
#         x0_2 = np.array([     0,     0,1.5e-4,     0,2.5e-5])
#         x0_3 = np.array([  7e-5,  1e-5,1.0e-4,  3e-5,  5e-6])
#         A = np.array([[1,0,2,1,0],[0,1,0,1,2]])

#         # All reactions here are binary reactions convert to different units
#         # and run the exact same job. Check that with unit conversion we get
#         # the same results each way.
#         x_1 = eq.sweep_titration(N, K, x0_1, units='M')
#         x_2 = eq.sweep_titration(N, K/1e3, 1e3*x0_2, units='mM')
#         x_3 = eq.sweep_titration(N, K/1e6, 1e6*x0_3, units='uM')
#         x_4 = eq.sweep_titration(N, K/1e9, 1e9*x0_3, units='nM')
#         x_5 = eq.sweep_titration(N, K/1e12, 1e12*x0_3, units='pM')

#         self.check_equilibrium(N, K, x_1)
#         self.check_equilibrium(N, K/1e3, x_2)

#         for xi1,xi2,xi3,xi4,xi5 in zip(x_1,x_2,x_3,x_4,x_5):
#             # convert back to normal units
#             self.assertEqualTolerance(xi1,xi2/1e3,TOL_DEC_PLACE)
#             self.assertEqualTolerance(xi1,xi3/1e6,TOL_DEC_PLACE)
#             self.assertEqualTolerance(xi1,xi4/1e9,TOL_DEC_PLACE)
#             self.assertEqualTolerance(xi1,xi5/1e12,TOL_DEC_PLACE)

#         # Check that the job gives the right results in the first place
#         self.check_equilibrium(N, K, x_1)
#         self.check_mass_conservation(A, x0_1, x_1)

#     def test_calculate_eqtk_non_canonical(self):
#         N = np.array([[-1, -1, -1, 1, 1, 0, 0],
#                       [-1,  0,  0, 0, 0,-1, 1],
#                       [-1,  1,  1, 0,-1, 0, 0],
#                       [ 0,  1, -1, 0,-1, 1, 0]])

#         K = np.array([1., 2., 3., 5.])
#         x0 = np.array([1., 2., 3., 5., 7., 11., 13.])

#         x = eq.sweep_titration(N, K, x0, units='micromolar')

#         self.check_equilibrium(N, K, x)

#         x0 = np.array([0., 0., 0., 0., 0., 0., 13.])

#         x = eq.sweep_titration(N, K, x0, units='micromolar')
#         self.check_equilibrium(N, K, x)

#         # TODO add any particular cases from the past that broke
#         # the algorithm here. Add any new ones we find.

#     def test_pruning_regression_1(self):
#         N = np.array([[-1,  0,  1,  0,  0,  0],[-1, -1,  0,  1,  0,  0],[ 0, -2,  0,  0,  1,  0],[ 0, -1, -1,  0,  0,  1]])
#         K = np.array([50.0, 10.0, 40.0, 100.0])
#         c_0 = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
#         c = eq.sweep_titration(N, K, c_0)
#         self.assertEqual(c[1] + c[3] + c[4] + c[5], 0)

#         trust_region_params = equ.TrustRegionParams(max_trial=1)
#         x, run_stats = equ.eqtk_conc_pure_python(
#             N, -np.log(K), c_0, trust_region_params=trust_region_params)

#         self.assertEqual(x[1] + x[3] + x[4] + x[5], 0)

#     def test_large_species(self):
#         # These currently fail. Uncomment to test them.
#         n_rxns = 30
#         N = np.diag(-2.0 * np.ones(n_rxns+1)) + np.diag(np.ones(n_rxns), 1)
#         N = N[:-1, :]
#         A = np.arange(0,n_rxns+1,1)
#         A = A.reshape((1,n_rxns+1))
#         A = 2**A

#         K = 100.0 * np.ones(n_rxns)
#         x0 = np.zeros(n_rxns + 1)
#         x0[0] = 1.0
#         x = eq.sweep_titration(N, K, x0, units='M', A=A, delta_bar=1e8)

#         # x = eq.sweep_titration(N, K, x0, units='M')

#     def test_pure_water(self):
#         N = np.array([[1, 1]])
#         K = np.array([0.00032571])
#         x0 = np.array([0, 0])

#         res = equ.eqtk_conc(N, -np.log(K), x0)
#         return self.assertTrue(res[1], "Pure water didn't converge")


# ###########################################################
# # Begin titration testing
# ###########################################################
#     def test_titration(self):
#         # Use default trust_region params, with one one trial
#         N = np.array([[-1,-1,1]])
#         K = np.array([1])
#         c0 = np.array([2,5,0.0])
#         initial_volume = 1.0
#         c0_titrant = np.array([1000, 0.0, 0.0])
#         vol_titrated = np.linspace(0,2e-3, 20)

#         x, run_stats = eq.volumetric_titration(
#             N=N, K=K, c_0=c0, initial_volume=initial_volume,
#             c_0_titrant=c0_titrant, vol_titrated=vol_titrated,
#             return_run_stats=True, units='uM',
#             max_trial=1)

#         A = np.array([[1,0,1],[0,1,1]])
#         for i,xi in enumerate(x):
#             self.check_equilibrium(N, K, xi)
#             x0_i = (c0_titrant * vol_titrated[i] + c0 * initial_volume) \
#                 / (vol_titrated[i] + initial_volume)
#             self.check_mass_conservation(A, xi, x0_i)

#         N = np.array([[-2,0,1,0,0],[-1,-1,0,1,0],[0,-2,0,0,1]])
#         K = np.array([2,3,8])
#         c0 = np.array([  3,  5,     0,     0,     0])
#         c0_titrant = np.array([0, 0, 0, 1000, 0])

#         A = np.array([[1,0,2,1,0],[0,1,0,1,2]])


#         x, run_stats = eq.volumetric_titration(
#             N=N, K=K, c_0=c0, initial_volume=initial_volume,
#             c_0_titrant=c0_titrant, vol_titrated=vol_titrated,
#             return_run_stats=True, units='mM',
#             tol = 1e-8,
#             max_trial=1)

#         for i,xi in enumerate(x):
#             self.check_equilibrium(N, K, xi)
#             x0_i = (c0_titrant * vol_titrated[i] + c0 * initial_volume) \
#                 / (vol_titrated[i] + initial_volume)
#             self.check_mass_conservation(A, xi, x0_i)


# class AGTests(ut.TestCase):
#     def assertEqualTolerance(self, a, b, sig_figs):
#         aa = abs(a)
#         ab = abs(b)
#         if aa == 0 or ab == 0:
#             self.assertTrue(aa < 10**(-sig_figs) and ab < 10**(-sig_figs))
#         else:
#             relative_diff = 2 * abs(a - b) / (aa + ab)
#             if relative_diff > 0:
#                 dec_place = - np.log(relative_diff) / np.log(10)
#             else:
#                 dec_place = sig_figs + 1.
#             self.assertTrue(
#                 dec_place >= sig_figs,msg \
#                     = "%.10le != %.10le agree to %lf of %i decimal places" \
#                     % (a,b,dec_place, sig_figs))

#     def check_equilibrium(self, N, K, x):
#         n_reacs = N.shape[0]
#         n_comps = N.shape[1]
#         if len(x.shape) == 1:
#             x = x.reshape((1, len(x)))
#         for xpt in x:
#             for k in range(n_reacs):
#                 eqval = 1
#                 for j in range(n_comps):
#                     if xpt[j] > 0:
#                         eqval *= xpt[j]**N[k,j]
#                     else:
#                         eqval = 0
#                 if eqval > 0:
#                     self.assertEqualTolerance(eqval, K[k], TOL_DEC_PLACE)

#     def test_overflow_tests(self):
#         import numpy as np
#         import eqtk_utils as equ

#         G = np.array([-2.280432e+02, -2.280432e+02, -1.040435e+03, 1.268478e+03 ])
#         A = np.array([[-0.66225429, 0.74096465, 0.07871036, 0.07871036],
#                                    [0.40177015,  0.22576844, 0.62753859, 0.62753859]])
#         x0 =  np.array([[1.80475639e-08, 4.51189097e-09, 0.00000000e+00, 0.00000000e+00]])

#         params = equ.TrustRegionParams(delta_bar=1000, max_iters=2000)
#         x, con, stats = equ.eqtk_conc_from_free_energies(A, G, x0, params)  # This will fail
#         self.assertTrue(con, msg="Failed to converge on overflow tests")

#         # Other stuff same, these G's also fail:
#         # G = np.array([-1.036404e+02, -1.036404e+02, 8.768368e+02, -7.731964e+02])
#         # G = np.array([3.136494e+02, 3.136494e+02, -7.871400e+02, 4.734906e+02])


#     def test_large_diff(self):
#         # Test situations where free energy differences are very large
#         A = [[1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#              [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
#              [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
#              [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
#         G = [0.0, 40, 40, 40, -2000, -2050]
#         x0 = [1e-6, 1e-6, 1e-6, 1.0000e-6, 0, 0]
#         x = eq.sweep_titration(N=None, K=None, A=A, G=G, c_0=x0, G_units=None, units=None)

#         A = [[1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#              [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
#              [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
#              [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
#         G = [0.0, 40, 40, 40, -2100, -2100]
#         x0 = [1e-6, 1e-6, 1e-6, 1.0001e-6, 0, 0]
#         x, run_stats = eq.sweep_titration(N=None, K=None, A=A, G=G, c_0=x0,
#                                 G_units=None, units=None,
#                                 maxtrial=1,
#                                 return_run_stats=True)

#         self.assertTrue(run_stats.n_trials == 1, msg="Required more than 1 trial")

#         A = [[1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
#              [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
#              [0.0, 0.0, 1.0, 0.0, 1.0, 1.0],
#              [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]]
#         G = [0.0, 40, 150, 200, -4200, -4215]
#         x0 = [1e-6, 1e-6, 1e-6, 1.0000e-6, 0, 0]
#         x, run_stats = eq.sweep_titration(
#             N=None, K=None, A=A, G=G, c_0=x0, G_units=None, units=None,
#             maxtrial=1, return_run_stats=True)
#         self.assertTrue(run_stats.n_trials == 1, msg="Required more than 1 trial")


#         # From an earlier failed case in the design algorithm.
#         A = [[1.0, 0.0, 1.0, 1.0, 0.0],
#              [1.0, 0.0, 1.0, 1.0, 0.0],
#              [1.0, 1.0, 0.0, 0.0, 2.0],
#              [1.0, 0.0, 0.0, 1.0, 2.0]]
#         G = [-1.1323012373599138e+02,
#              -2.7028447814426110e-01,
#              -2.3382656193096754e+01,
#              -1.0088531260804201e+02,
#              -5.7676558386243052e+01]
#         x0 = [1.8134373707286439e-08,
#               3.5913242229740680e-14,
#               3.5913242229740680e-14,
#               3.5913242229740680e-14,
#               1.7956621114870340e-14]

#         eq.sweep_titration(N=None, K=None, A=A, G=G, c_0=x0, G_units=None, units=None,
#                 maxtrial=1, return_run_stats=True)

#     def test_large(self):
#         A = [[1.0, 1.0, 0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, \
#               0.0, 0.0, 0.0, 3.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
#               1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
#               0.0, 0.0, 0.0, 4.0, 3.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, \
#               2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
#               1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
#               1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, \
#               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, \
#               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
#              [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, \
#               0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, \
#               0.0, 1.0, 0.0, 0.0, 3.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, \
#               0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, \
#               0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 3.0, 2.0, 2.0, 2.0, 1.0, \
#               2.0, 1.0, 1.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, \
#               0.0, 0.0, 0.0, 2.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
#               4.0, 3.0, 3.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, \
#               1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],\
#              [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, \
#               2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, \
#               1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 3.0, \
#               2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, \
#               1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, \
#               0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, \
#               2.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.0, \
#               0.0, 1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 2.0, 1.0, 3.0, 2.0, 2.0, \
#               1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 4.0, 3.0, 2.0, 2.0, 1.0, 0.0],\
#              [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, \
#               0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
#               1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0, \
#               1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
#               1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, \
#               1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, \
#               1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, \
#               0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 1.0, 0.0, 1.0, 1.0, \
#               2.0, 2.0, 1.0, 2.0, 2.0, 3.0, 0.0, 1.0, 2.0, 2.0, 3.0, 4.0]]
#         G = [-4.2015367534032657e+02, -1.2703232169812192e+02,
#              -5.4817682865924418e+01, -7.8102170476565016e+00,
#              -5.0193490507510681e+01, -2.7165182426095953e+02,
#              -2.2879300425908158e+02, -1.4323976440450812e+02,
#              -1.8958975769447511e+02, -1.4124781488205676e+02,
#              -8.6673976923205487e+01, -2.0133236499415651e+02,
#              -3.2653642782462079e+01, -6.7053699416077549e+01,
#              -1.1963351823998251e+02, -4.1489422124390160e+02,
#              -3.7341204386611190e+02, -2.8870092306527079e+02,
#              -3.3220415271165064e+02, -2.9702434984420739e+02,
#              -2.6053825202585415e+02, -3.8323486406656087e+02,
#              -2.3501213478769949e+02, -1.6877304869504263e+02,
#              -2.0824653925915831e+02, -3.3590996661662558e+02,
#              -2.0520012341497807e+02, -2.5971699676849943e+02,
#              -2.1094947752425242e+02, -1.7385165483988780e+02,
#              -2.8773765389188344e+02, -1.0538997980893089e+02,
#              -2.3825117626792220e+02, -2.2061630240883488e+02,
#              -2.6763222147566859e+02, -4.4752134398923594e+01,
#              -9.1402458513709178e+01, -1.3718133395724152e+02,
#              -1.8524123876379437e+02, -5.5865029666144483e+02,
#              -5.1489457397372496e+02, -4.3343826706414734e+02,
#              -4.7510981508132687e+02, -4.5612936186968261e+02,
#              -4.0516544010432943e+02, -5.2786658540528890e+02,
#              -3.7963121373685630e+02, -3.1423591441078139e+02,
#              -3.5051515988808501e+02, -4.8060673985504400e+02,
#              -3.4836654223951467e+02, -4.0233100683437743e+02,
#              -4.7197688702079347e+02, -3.9381231460595689e+02,
#              -5.1781246568902998e+02, -3.8326654234796888e+02,
#              -3.2915592514831127e+02, -4.4917341152776635e+02,
#              -3.2983630637315213e+02, -2.7921350085332392e+02,
#              -4.4497398982119728e+02, -3.9412684034275190e+02,
#              -4.4928628483319653e+02, -3.0552056600426687e+02,
#              -3.4915651913700128e+02, -3.1294557706150187e+02,
#              -2.6675738226782761e+02, -3.8945399430852598e+02,
#              -2.6054781451352403e+02, -1.8459981759035912e+02,
#              -2.3377960947381828e+02, -3.7152431456702061e+02,
#              -2.2687638301116129e+02, -2.7837400045836358e+02,
#              -3.9842931333298338e+02, -4.2796378507687069e+02,
#              -3.6698879215177789e+02, -4.3182337303644073e+02,
#              -3.5519390403130399e+02, -2.3065815130972845e+02,
#              -2.7657722516245099e+02, -4.0100240632810880e+02,
#              -2.7532513340609239e+02, -3.2497380477812760e+02,
#              -2.9620693634022933e+02, -2.4460174120284984e+02,
#              -3.5701620136122949e+02, -1.9255737553808018e+02,
#              -3.2465646516596814e+02, -3.0321729265725014e+02,
#              -4.1421271587576661e+02, -2.0506920063988736e+02,
#              -3.1964834666572102e+02, -1.2588463387721514e+02,
#              -2.5691105508367849e+02, -2.5753511368260058e+02,
#              -3.0307716707347032e+02, -4.3284119923642481e+02,
#              -2.4211326870787389e+02, -2.9484362717015421e+02,
#              -2.8668170416631432e+02, -3.3775513541674047e+02,
#              -6.9179245072223310e+01, -1.0595181053210460e+02,
#              -1.6153764121002649e+02, -1.5335633483058461e+02,
#              -2.0267567033456882e+02, -2.5467945122971807e+02]
#         x0 = [1.8134373707286439e-08, 3.5913242229740680e-14,
#               3.5913242229740680e-14, 3.5913242229740680e-14,
#               3.5913242229740680e-14, 1.7956621114870340e-14,
#               3.5913242229740680e-14, 3.5913242229740680e-14,
#               3.5913242229740680e-14, 1.7956621114870340e-14,
#               3.5913242229740680e-14, 3.5913242229740680e-14,
#               1.7956621114870340e-14, 3.5913242229740680e-14,
#               1.7956621114870340e-14, 1.1971080743246893e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               3.5913242229740680e-14, 3.5913242229740680e-14,
#               3.5913242229740680e-14, 1.7956621114870340e-14,
#               3.5913242229740680e-14, 3.5913242229740680e-14,
#               3.5913242229740680e-14, 1.7956621114870340e-14,
#               1.1971080743246893e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               3.5913242229740680e-14, 3.5913242229740680e-14,
#               1.7956621114870340e-14, 1.1971080743246893e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.1971080743246893e-14, 8.9783105574351699e-15,
#               1.1971080743246893e-14, 1.1971080743246893e-14,
#               1.1971080743246893e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.1971080743246893e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 3.5913242229740680e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 3.5913242229740680e-14,
#               1.7956621114870340e-14, 1.1971080743246893e-14,
#               1.7956621114870340e-14, 3.5913242229740680e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               3.5913242229740680e-14, 1.7956621114870340e-14,
#               3.5913242229740680e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.1971080743246893e-14,
#               8.9783105574351699e-15, 1.1971080743246893e-14,
#               1.1971080743246893e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.1971080743246893e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.7956621114870340e-14, 1.1971080743246893e-14,
#               8.9783105574351699e-15, 1.1971080743246893e-14,
#               1.7956621114870340e-14, 1.7956621114870340e-14,
#               1.1971080743246893e-14, 8.9783105574351699e-15]
#         x = eq.sweep_titration(
#             N=None, K=None, A=A, G=G, c_0=x0, G_units=None, units=None, maxtrial=1)


# if __name__ == '__main__':
#     ut.main()
