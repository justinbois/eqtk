import pytest

import numpy as np
import eqtk


def test_promiscuous_binding_failure():
    A = np.array(
        [
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            [
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
                0.0,
                0.0,
                1.0,
                1.0,
            ],
        ]
    )
    G = np.array(
        [
            -0.51720535,
            -0.69471304,
            -1.78260496,
            -1.32337777,
            -0.63267947,
            -0.57923893,
            -0.78718634,
            -0.27521037,
            -0.13733511,
            -0.69433251,
            1.6858364,
            -0.43683479,
            0.39312096,
            -0.0625205,
            0.23139303,
            0.07680628,
            -0.52774543,
            1.74592678,
        ]
    )
    x0 = np.array(
        [
            [
                2.48257788e01,
                1.72132293e-01,
                1.14833731e-02,
                5.00547317e-02,
                1.38949549e-01,
                1.93069773e01,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ]
        ]
    )


def test_spontaneous_production_failure():
    N = np.array(
        [[1, 0, 1, 0, -1, 0], [1, 0, 0, 1, 0, -1], [1, 1, 1, 0, 0, 0]], dtype=float
    )

    A = np.array(
        [[0, 0, 0, 1, 0, 1], [1, 0, -1, 0, 0, 1], [0, -1, 1, 0, 1, 0]], dtype=float
    )

    G = np.array([0, 1, 2, 3, 4, 5])
    K = np.exp(-np.dot(N, G))

    for x0_val in [
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]:
        x0 = np.array(x0_val, dtype=float)
        x_NK = eqtk.solve(c0=x0, N=N, K=K)

        with pytest.raises(ValueError) as excinfo:
            x_AG = eqtk.solve(c0=x0, A=A, G=G)
        excinfo.match("`A` must have all nonnegative entries.")

        assert eqtk.eqcheck(x_NK, x0, N=N, K=K)


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
    x = eqtk.solve(c0=x0, A=A, G=G)
    assert eqtk.eqcheck(x, x0, A=A, G=G)


def test_trivial_elemental_failure():
    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    G = np.array([0.0, 0.0])
    x0 = np.array([[3.48219906e-06, 1.32719868e-10]])
    assert np.allclose(eqtk.solve(c0=x0, A=A, G=G), x0)

    A = np.array([[1.0, 0.0], [0.0, 1.0]])
    G = np.array([0.0, 0.0])
    x0 = np.array([[2.24222410e-08, 1.63359284e-04]])
    assert np.allclose(eqtk.solve(c0=x0, A=A, G=G), x0)

    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    G = np.array([0.0, 0.0, 0.0])
    x0 = np.array([[2.63761955e-04, 4.93360042e-07, 4.88340687e-07]])
    assert np.allclose(eqtk.solve(c0=x0, A=A, G=G), x0)


def test_past_failure_1():
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
    x = eqtk.solve(x0, A=A, G=G)
    assert eqtk.eqcheck(x, x0, A=A, G=G)


def test_past_failure_2():
    N = np.array([[-2.0, 1.0, 0.0, 0.0], [-3.0, 0.0, 1.0, 0.0], [-4.0, 0.0, 0.0, 1.0]])
    minus_log_K = np.array([-43.66660344, -68.14676841, -92.28023823])
    x0 = np.array([[1.87852623e-06, 3.75705246e-06, 1.25235082e-06, 4.69631557e-07]])
    K = np.exp(-minus_log_K)
    x = eqtk.solve(x0, N, K)
    assert eqtk.eqcheck(x, x0, N, K)


def test_small_conc_failure():
    A = np.array(
        [
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 1.0, 2.0],
        ]
    )
    G = np.array(
        [
            -1.1323012373599138e02,
            -2.7028447814426110e-01,
            -2.3382656193096754e01,
            -1.0088531260804201e02,
            -5.7676558386243052e01,
        ]
    )
    x0 = np.array(
        [
            [
                1.8134373707286439e-08,
                3.5913242229740680e-14,
                3.5913242229740680e-14,
                3.5913242229740680e-14,
                1.7956621114870340e-14,
            ]
        ]
    )
    x = eqtk.solve(c0=x0, A=A, G=G)
    assert eqtk.eqcheck(x, x0, A=A, G=G)
