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
        eqtk.checks.check_input([], N, K, None, None, *tuple([None]*4))
    excinfo.match("c0 and N must have same number of columns.")

    with pytest.raises(ValueError) as excinfo:
        eqtk.checks.check_input([0.1, 0.2, 0.1, 0.2, 0.3], N, K, None, None, *tuple([None]*4))
    excinfo.match("c0 and N must have same number of columns.")

    with pytest.raises(ValueError) as excinfo:
        eqtk.checks.check_input([], None, None, A, G, *tuple([None]*4))
    excinfo.match("c0 and A must have same number of columns.")

    with pytest.raises(ValueError) as excinfo:
        eqtk.checks.check_input([0.1, 0.2, 0.1, 0.2, 0.3], None, None, A, G, *tuple([None]*4))
    excinfo.match("c0 and A must have same number of columns.")

    # Should raise no exception
    _ = eqtk.checks.check_input(
        [0.1, 0.2, 0.1, 0.2, 0.3, 0.1], N, K, None, None, *tuple([None]*4)
    )
    _ = eqtk.checks.check_input(
        [0.1, 0.2, 0.1, 0.2, 0.3, 0.1], None, None, A, G, *tuple([None]*4)
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
    target = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    c0 = [1, 2, 3, 4, 5, 6]
    c0, _, _, _, _ = eqtk.checks.check_input(c0, N, K, None, None, *tuple([None]*4))
    assert np.array_equal(c0, target)

    c0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    c0, _, _, _, _ = eqtk.checks.check_input(c0, N, K, None, None, *tuple([None]*4))
    assert np.array_equal(c0, target)

    c0 = np.array([[1, 2, 3, 4, 5, 6]])
    c0, _, _, _, _ = eqtk.checks.check_input(c0, N, K, None, None, *tuple([None]*4))
    assert np.array_equal(c0, target)

    c0 = [[1, 2, 3, 4, 5, 6]]
    c0, _, _, _, _ = eqtk.checks.check_input(c0, N, K, None, None, *tuple([None]*4))
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
        eqtk.checks.check_input([0.1, 0.2, 0.1, 0.2, 0.3], None, None, A, G, *tuple([None]*4))
    excinfo.match("A must have full row rank.")


def test_reshape_empty_A():
    A = np.array([[]]).reshape((1, 0)).astype(float)
    G = np.array([1.0])
    x0 = np.array([1.0])
    x0, N, K, A, G = eqtk.checks.check_input(x0, None, None, A, G, *tuple([None]*4))
    assert A.shape[0] == 0
    assert A.shape[1] == 1


def test_A_negative():
    A = np.array([[1, -1]])
    G = np.ones(2)
    with pytest.raises(ValueError) as excinfo:
        eqtk.checks.check_input([], None, None, A, G, *tuple([None]*4))
    excinfo.match("A must have all nonnegative entries.")

    A = np.array([[0, 0, 1, 1], [1, -1, 0, 1]])
    G = np.ones(4)
    with pytest.raises(ValueError) as excinfo:
        eqtk.checks.check_input([], None, None, A, G, *tuple([None]*4))
    excinfo.match("A must have all nonnegative entries.")
