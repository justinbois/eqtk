import itertools

import numpy as np
import eqtk

def test_boolean_index_1d():
    a = np.array([1., 2., 3., 4.])
    for t in itertools.product(*([True, False] for _ in range(len(a)))):
        b = np.array(t)
        assert np.array_equal(eqtk.eqtk._boolean_index(a, b, np.sum(b)), a[b])

def test_boolean_index_2d():
    a = np.array([[1., 2., 3., 4., 5., 6.], 
                  [7., 8., 9., 10., 11., 12.], 
                  [13., 14., 15., 16., 17., 18.], 
                  [19., 20., 21., 22., 23., 24.]])
    for t_row in itertools.product(*([True, False] for _ in range(a.shape[0]))):
        b_row = np.array(t_row)
        n_true_row = np.sum(b_row)
        for t_col in itertools.product(*([True, False] for _ in range(a.shape[1]))):
            b_col = np.array(t_col)
            n_true_col = np.sum(b_col)

        target = a[b_row, :]
        target = target[:, b_col]

        assert np.array_equal(eqtk.eqtk._boolean_index_2d(a, b_row, b_col, n_true_row, n_true_col), target)
