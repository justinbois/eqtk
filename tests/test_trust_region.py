import numpy as np
import eqtk.trust_region

import testcases

import hypothesis
import hypothesis.strategies as hs
import hypothesis.extra.numpy as hnp

# 1D arrays for testing functions outside of edge cases
array_shapes = hnp.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=10)
arrays = hnp.arrays(np.float, array_shapes, elements=hs.floats(-100, 100))

# 2D matrices
array_shapes_2d = hnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=10)
arrays_2d = hnp.arrays(np.float, array_shapes_2d, elements=hs.floats(-100, 100))

# Nonsquare 2D matrix (hack to get matrix and vector with correct dimensiones)
matvec_shapes_2d = hnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=10)
matvec_2d = hnp.arrays(np.float, array_shapes_2d, elements=hs.floats(-100, 100))


def random_pos_def(n):
    """Generates a nxn random positive definite matrix."""
    L = np.tril(random_matrix(n))
    return np.dot(L, L.transpose())


def random_matrix(n):
    return np.random.uniform(low=-1, high=1, size=(n, n))


def random_array(n):
    return np.random.uniform(low=-1, high=1, size=n)


@hypothesis.settings(deadline=None)
@hypothesis.given(hs.integers(min_value=1, max_value=10))
def test_rescale_hes(n):
    B = random_pos_def(n)
    d = eqtk.trust_region.inv_scaling(B)
    B_rescaled = eqtk.trust_region.scaled_hes(B, d)
    assert np.allclose(np.diag(B_rescaled), np.ones(n))
    assert np.allclose(np.diag(1 / d) @ B_rescaled @ np.diag(1 / d), B)


def test_no_cholesky_failures(
    n_random_test_cases=200, max_particles=4, max_compound_size=5
):
    # Generate list of random test cases
    random_test_cases = []
    for i in range(n_random_test_cases):
        n_parts = np.random.randint(1, max_particles)
        max_cmp_size = np.random.randint(1, max_compound_size + 1)
        random_test_cases.append(
            testcases.random_elemental_test_case(n_parts, max_cmp_size)
        )

    for tc in random_test_cases:
        conserv_vector = np.dot(tc["A"], tc["c0"])
        logx, converged, n_trial, step_tally = eqtk.eqtk._solve_trust_region(
            tc["A"], tc["G"], conserv_vector
        )
        assert converged
        assert n_trial == 1
        assert (
            step_tally[3] == 0
        ), f"{step_tally[3]} Cholesky failures forcing a Cauchy step"
        assert step_tally[4] == 0, f"{step_tally[4]} irrelevant Cholesky failures"
        assert step_tally[5] == 0, f"{step_tally[5]} dogleg failures"
