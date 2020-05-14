import numpy as np

import eqtk.solvers
import eqtk.linalg


def test_positive_A():
    tol = 1e-7
    tol_zero = 1e-12

    A = np.array([[1, 0, 0, 1, 1], [0, 1, 0, 1, 0], [0, 0, 1, 0, 1]], dtype=float)
    x0 = np.array([1, 1, 1, 0, 0], dtype=float)
    abs_tol_target = np.array([1e-7, 1e-7, 1e-7], dtype=float)
    abs_tol = eqtk.solvers._tolerance(tol, tol_zero, A, x0)
    assert np.allclose(abs_tol, abs_tol_target)

    x0 = np.array([0.001, 0.002, 0.003, 0, 0], dtype=float)
    abs_tol_target = np.array([1e-10, 2e-10, 3e-10], dtype=float)
    abs_tol = eqtk.solvers._tolerance(tol, tol_zero, A, x0)
    assert np.allclose(abs_tol, abs_tol_target)

    A = np.array([[1, 0, 0, 1, 3], [0, 1, 0, 2, 0], [0, 0, 1, 0, 4]], dtype=float)
    x0 = np.array([1, 1, 1, 0, 0], dtype=float)
    abs_tol_target = np.array([1e-7, 1e-7, 1e-7], dtype=float)
    abs_tol = eqtk.solvers._tolerance(tol, tol_zero, A, x0)
    assert np.allclose(abs_tol, abs_tol_target)

    A = np.array([[1, 0, 0, 1, 3], [0, 2, 0, 2, 0], [0, 0, 3, 0, 4]], dtype=float)
    x0 = np.array([1, 1, 1, 1, 1], dtype=float)
    abs_tol_target = np.array([5e-7, 4e-7, 7e-7], dtype=float)
    abs_tol = eqtk.solvers._tolerance(tol, tol_zero, A, x0)
    assert np.allclose(abs_tol, abs_tol_target)


def test_cancelling_entries():
    tol = 1e-7
    tol_zero = 1e-12

    A = np.array([[1, -1]], dtype=float)
    x0 = np.array([1, 1], dtype=float)
    abs_tol_target = np.array([1e-7], dtype=float)
    abs_tol = eqtk.solvers._tolerance(tol, tol_zero, A, x0)
    assert np.allclose(abs_tol, abs_tol_target)

    A = np.array([[1, -1]], dtype=float)
    x0 = np.array([1, 2], dtype=float)
    abs_tol_target = np.array([2e-7], dtype=float)
    abs_tol = eqtk.solvers._tolerance(tol, tol_zero, A, x0)
    assert np.allclose(abs_tol, abs_tol_target)

    A = np.array([[1, 0, 1, 0]], dtype=float)
    x0 = np.array([0, 1, 0, 1], dtype=float)
    abs_tol_target = np.array([tol_zero], dtype=float)
    abs_tol = eqtk.solvers._tolerance(tol, tol_zero, A, x0)
    assert np.allclose(abs_tol, abs_tol_target)


def test_zero_conc():
    tol = 1e-7
    tol_zero = 1e-12

    A = np.array([[1, 1]], dtype=float)
    x0 = np.array([0, 0], dtype=float)
    abs_tol_target = np.array([tol_zero], dtype=float)
    abs_tol = eqtk.solvers._tolerance(tol, tol_zero, A, x0)
    assert np.allclose(abs_tol, abs_tol_target)

    A = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    x0 = np.array([0, 0, 0], dtype=float)
    abs_tol_target = np.array([tol_zero], dtype=float)
    abs_tol = eqtk.solvers._tolerance(tol, tol_zero, A, x0)
    assert np.allclose(abs_tol, abs_tol_target)
