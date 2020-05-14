import itertools

import numpy as np
import eqtk


def test_boolean_index_1d():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    for t in itertools.product(*([True, False] for _ in range(len(a)))):
        b = np.array(t)
        assert np.array_equal(eqtk.solvers._boolean_index(a, b, np.sum(b)), a[b])


def test_boolean_index_2d():
    a = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
            [19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
        ]
    )
    for t_row in itertools.product(*([True, False] for _ in range(a.shape[0]))):
        b_row = np.array(t_row)
        n_true_row = np.sum(b_row)
        for t_col in itertools.product(*([True, False] for _ in range(a.shape[1]))):
            b_col = np.array(t_col)
            n_true_col = np.sum(b_col)

        target = a[b_row, :]
        target = target[:, b_col]

        assert np.array_equal(
            eqtk.solvers._boolean_index_2d(a, b_row, b_col, n_true_row, n_true_col),
            target,
        )
