import numpy as np
import eqtk.trust_region

import hypothesis
import hypothesis.strategies as hs
import hypothesis.extra.numpy as hnp

# 1D arrays for testing functions outside of edge cases
array_shapes = hnp.array_shapes(min_dims=1, max_dims=1, min_side=2, max_side=10)
arrays = hnp.arrays(np.double, array_shapes, elements=hs.floats(-100, 100))

# 2D matrices
array_shapes_2d = hnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=10)
arrays_2d = hnp.arrays(np.double, array_shapes_2d, elements=hs.floats(-100, 100))

# Nonsquare 2D matrix (hack to get matrix and vector with correct dimensiones)
matvec_shapes_2d = hnp.array_shapes(min_dims=2, max_dims=2, min_side=2, max_side=10)
matvec_2d = hnp.arrays(np.double, array_shapes_2d, elements=hs.floats(-100, 100))


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
