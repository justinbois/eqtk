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
    _ = eqtk.checks.check_input([0.1, 0.2, 0.1, 0.2, 0.3, 0.1], N=N, K=K, A=None, G=None)
    _ = eqtk.checks.check_input([0.1, 0.2, 0.1, 0.2, 0.3, 0.1], A=A, G=G, N=None, K=None)
