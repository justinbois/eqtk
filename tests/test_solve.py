import numpy as np
import eqtk

import hypothesis
import hypothesis.strategies as hs
import hypothesis.extra.numpy as hnp

# 1D arrays
array_shapes = hnp.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=10)
arrays = hnp.arrays(np.double, array_shapes, elements=hs.floats(-100, 100))

# 2D matrices
array_shapes_2d = hnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=10)
arrays_2d = hnp.arrays(np.double, array_shapes_2d, elements=hs.floats(-100, 100))


@hypothesis.settings(
    deadline=None,
    max_examples=500,
    suppress_health_check=[hypothesis.HealthCheck.filter_too_much],
)
@hypothesis.given(arrays_2d)
def test_NK_formulation(N):
    hypothesis.assume(not np.isnan(N).any())
    hypothesis.assume(np.linalg.matrix_rank(N) == N.shape[0])

    hp_x0 = hnp.arrays(np.double, (N.shape[1],), elements=hs.floats(0, 1))
    hp_K = hnp.arrays(np.double, (N.shape[0],), elements=hs.floats(1e-16, 1e10))

    @hypothesis.settings(
        deadline=None, suppress_health_check=[hypothesis.HealthCheck.filter_too_much]
    )
    @hypothesis.given(hp_K, hp_x0)
    def compute_eq(K, x0):
        hypothesis.assume(not (np.isnan(x0).any() or np.isnan(K).any()))
        x = eqtk.solve(x0, N=N, K=K)

        eq_check, cons_check = eqtk.eqcheck(x, x0, N=N, K=K)
        assert np.isclose(eq_check, 1.0).all(), "Equilibrium error"
        assert np.isclose(cons_check, 1.0).all(), "Conservation of mass error"


@hypothesis.settings(
    deadline=None,
    max_examples=500,
    suppress_health_check=[hypothesis.HealthCheck.filter_too_much],
)
@hypothesis.given(arrays_2d)
def test_AG_formulation(A):
    hypothesis.assume(not np.isnan(A).any())
    hypothesis.assume(np.linalg.matrix_rank(A) == A.shape[0])

    hp_x0 = hnp.arrays(np.double, (A.shape[1],), elements=hs.floats(0, 1))
    hp_G = hnp.arrays(np.double, (A.shape[0],), elements=hs.floats(-100, 100))

    @hypothesis.settings(
        deadline=None, suppress_health_check=[hypothesis.HealthCheck.filter_too_much]
    )
    @hypothesis.given(hp_G, hp_x0)
    def compute_eq(G, x0):
        hypothesis.assume(not (np.isnan(x0).any() or np.isnan(G).any()))
        x = eqtk.solve(x0, A=A, G=G)

        eq_check, cons_check = eqtk.eqcheck(x, x0, A=A, G=G)
        assert np.isclose(eq_check, 1.0).all(), "Equilibrium error"
        assert np.isclose(cons_check, 1.0).all(), "Conservation of mass error"
