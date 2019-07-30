import numpy as np


def numba_check():
    """Check to see if numba is available and working properly.

    Returns
    -------
    output : bool
        True is numba is installed and working properly; False
        otherwise.
    """
    try:
        import numba

        if numba.__version__ >= "0.42.0":
            have_numba = True

            # Sometimes linalg things fail with inconsistent BLAS installations
            # Check that.
            A = np.array([[0.06, 0.2], [0.2, 0.7]])
            b = np.array([1.2, 0.3])

            @numba.njit
            def my_solve(A, b):
                return np.linalg.solve(A, b)

            x = my_solve(A, b)

        else:
            have_numba = False
    except:
        have_numba = False

    return have_numba
