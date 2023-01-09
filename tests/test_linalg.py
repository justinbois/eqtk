import numpy as np
import eqtk.linalg
import eqtk.constants

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


def permutation_matrix(p):
    P = np.zeros((len(p), len(p)))
    for i, j in enumerate(p):
        P[i, j] = 1
    return P


@hypothesis.settings(deadline=None)
@hypothesis.given(hs.integers(min_value=1, max_value=10))
def test_diag_multiply(n):
    A = random_matrix(n)
    d = random_array(n)
    correct = np.diag(d) @ A @ np.diag(d)
    assert np.allclose(correct, eqtk.linalg.diag_multiply(A, d))


@hypothesis.settings(deadline=None)
@hypothesis.given(hs.integers(min_value=1, max_value=10))
def test_modified_cholesky(n):
    A = random_pos_def(n)
    L, p, success = eqtk.linalg.modified_cholesky(A)
    P = permutation_matrix(p)
    assert np.allclose(L @ L.transpose(), P @ A @ P.transpose())


@hypothesis.settings(deadline=None)
@hypothesis.given(hs.integers(min_value=1, max_value=10))
def test_lower_tri_solve(n):
    A = random_pos_def(n)
    b = random_array(n)
    L = np.tril(A)
    x = eqtk.linalg.lower_tri_solve(L, b)
    assert np.allclose(L @ x, b)


@hypothesis.settings(deadline=None)
@hypothesis.given(hs.integers(min_value=1, max_value=10))
def test_upper_tri_solve(n):
    A = random_pos_def(n)
    b = random_array(n)
    U = np.triu(A)
    x = eqtk.linalg.upper_tri_solve(U, b)
    assert np.allclose(U @ x, b)


@hypothesis.settings(deadline=None)
@hypothesis.given(hs.integers(min_value=1, max_value=10))
def test_modified_cholesky_solve(n):
    L = np.tril(random_matrix(n))
    A = L @ L.transpose()
    b = random_array(n)
    p = np.arange(n)
    assert np.allclose(A @ eqtk.linalg.modified_cholesky_solve(L, p, b), b)


@hypothesis.settings(deadline=None)
@hypothesis.given(hs.integers(min_value=1, max_value=10))
def test_solve_pos_def(n):
    A = random_pos_def(n)
    b = random_array(n)
    x, success = eqtk.linalg.solve_pos_def(A, b)
    assert success
    assert np.allclose(A @ x, b)


@hypothesis.settings(deadline=None)
@hypothesis.given(arrays_2d)
def test_nullspace_svd(N):
    assert np.allclose(
        np.dot(
            N, eqtk.linalg.nullspace_svd(N, eqtk.constants._nullspace_tol).transpose()
        ),
        0,
    )


def test_previous_nullspace_rank_failure():
    # This one had a rank failure
    A = np.array([[1, 1, 0, -1, 0], [1, 0, 1, 0, -1]]).astype(float)
    assert np.allclose(
        np.dot(
            A, eqtk.linalg.nullspace_svd(A, eqtk.constants._nullspace_tol).transpose()
        ),
        0,
    )
