import numpy as np


def numba_check():
    """Check to see if numba is available and working properly.

    Returns
    -------
    functioning_numba : bool
        True if numba is installed and working properly; False
        otherwise.
    jit : function
        Function used in `@jit` decorators. If `numba` is not available,
        a dummy function is returned. Otherwise, `numba.jit()` is
        returned.
    """
    # If you want to test without Numba, uncomment the following line.
    # Also useful for building docs because autodoc doesn't like Numba'd functions.
    # return False, _dummy_jit

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

    if have_numba:
        jit = numba.jit
    else:
        jit = _dummy_jit

    return have_numba, jit


def _dummy_jit(*args, **kwargs):
    """Dummy wrapper for jitting if numba is not installed."""

    def wrapper(f):
        return f

    def marker(*args, **kwargs):
        return marker

    if (
        len(args) > 0
        and (args[0] is marker or not callable(args[0]))
        or len(kwargs) > 0
    ):
        # @jit(int32(int32, int32)), @jit(signature="void(int32)")
        return wrapper
    elif len(args) == 0:
        # @jit()
        return wrapper
    else:
        # @jit
        return args[0]
