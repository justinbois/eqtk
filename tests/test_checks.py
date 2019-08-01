import numpy as np
import pytest

import eqtk.checks


def test_c0_mismatch():
    N = np.array(
        [
            [-1, 0, 1, 0, 0, 0],
            [-1, -1, 0, 1, 0, 0],
            [0, -2, 0, 0, 1, 0],
            [0, -1, -1, 0, 0, 1],
        ]
    )
    K = np.array([0.001, 0.002, 0.003, 0.004])
    A = np.array([[1, 0, 1, 1, 0, 1], [0, 1, 0, 1, 2, 1]])
    G = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006])

    with pytest.raises(ValueError) as excinfo:
        eqtk.checks.check_input([], N=N, K=K, A=None, G=None)
    excinfo.match("c0 and N must have same number of columns.")

    with pytest.raises(ValueError) as excinfo:
        eqtk.checks.check_input([0.1, 0.2, 0.1, 0.2, 0.3], N=N, K=K, A=None, G=None)
    excinfo.match("c0 and N must have same number of columns.")

    with pytest.raises(ValueError) as excinfo:
        eqtk.checks.check_input([], A=A, G=G, N=None, K=None)
    excinfo.match("c0 and A must have same number of columns.")

    with pytest.raises(ValueError) as excinfo:
        eqtk.checks.check_input([0.1, 0.2, 0.1, 0.2, 0.3], A=A, G=G, N=None, K=None)
    excinfo.match("c0 and A must have same number of columns.")

    # Should raise no exception
    _ = eqtk.checks.check_input(
        [0.1, 0.2, 0.1, 0.2, 0.3, 0.1], N=N, K=K, A=None, G=None
    )
    _ = eqtk.checks.check_input(
        [0.1, 0.2, 0.1, 0.2, 0.3, 0.1], A=A, G=G, N=None, K=None
    )


def test_c0_conversions():
    N = np.array(
        [
            [-1, 0, 1, 0, 0, 0],
            [-1, -1, 0, 1, 0, 0],
            [0, -2, 0, 0, 1, 0],
            [0, -1, -1, 0, 0, 1],
        ]
    )
    K = np.array([0.001, 0.002, 0.003, 0.004])
    target = np.array([[1., 2., 3., 4., 5., 6.]])

    c0 = [1, 2, 3, 4, 5, 6]
    c0, _, _, _, _ = eqtk.checks.check_input(c0, N, K, None, None)
    assert np.array_equal(c0, target)

    c0 = np.array([1., 2., 3., 4., 5., 6.])
    c0, _, _, _, _ = eqtk.checks.check_input(c0, N, K, None, None)
    assert np.array_equal(c0, target)

    c0 = np.array([[1, 2, 3, 4, 5, 6]])
    c0, _, _, _, _ = eqtk.checks.check_input(c0, N, K, None, None)
    assert np.array_equal(c0, target)

    c0 = [[1, 2, 3, 4, 5, 6]]
    c0, _, _, _, _ = eqtk.checks.check_input(c0, N, K, None, None)
    assert np.array_equal(c0, target)


def test_A_rank_deficient():
    A = np.array(
        [
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 1.0, 2.0],
        ]
    )
    G = np.zeros(5)
    with pytest.raises(ValueError) as excinfo:
        eqtk.checks.check_input([0.1, 0.2, 0.1, 0.2, 0.3], A=A, G=G, N=None, K=None)
    excinfo.match("A must have full row rank.")
