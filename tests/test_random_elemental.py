import random
import numpy as np
import eqtk
import eqtk.testcases

import hypothesis
import hypothesis.strategies as hs
import hypothesis.extra.numpy as hnp

# 1D arrays
array_shapes = hnp.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=10)
arrays = hnp.arrays(np.double, array_shapes, elements=hs.floats(-100, 100))

# 2D matrices
array_shapes_2d = hnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=10)
arrays_2d = hnp.arrays(np.double, array_shapes_2d, elements=hs.floats(-100, 100))


def test_random_cases(n_random_test_cases=200, max_particles=4, max_compound_size=5):
    # Generate list of random test cases
    random_test_cases = []
    for i in range(n_random_test_cases):
        n_parts = np.random.randint(1, max_particles)
        max_cmp_size = np.random.randint(1, max_compound_size + 1)
        random_test_cases.append(
            eqtk.testcases.random_elemental_test_case(n_parts, max_cmp_size)
        )

    for tc in random_test_cases:
        c_NK = eqtk.solve(c0=tc["c0"], N=tc["N"], K=tc["K"])
        c_AG = eqtk.solve(c0=tc["c0"], A=tc["A"], G=tc["G"])
        assert np.allclose(c_NK, tc["c"])
        assert np.allclose(c_AG, tc["c"])
        assert eqtk.eqcheck(c_NK, tc["c0"], tc["N"], tc["K"])
        assert eqtk.eqcheck(c_AG, tc["c0"], A=tc["A"], G=tc["G"])


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
