import numpy as np

from . import numba_check

have_numba, jit = numba_check.numba_check()


@jit(nopython=True)
def lup_decomposition(A_in):

    """

    Performs LUP decomposition of square matrix A.
    The LUP decomposition is P.A = L.U
    The lower triangle A[i][j]: i > j is set to L[i][j]
    The diagonal of L is always unity
    The upper triangle including the diagonal: A[i][j]: i <= j is set to U[i][j]
    p[i] = j: P[i][j] = 1 in the above expression
    A should be n x n, p should be n elements.
    This algorithm is based on Algorithm 3.4.1 in Golub and van Loan with
    extension from the end of 3.4.4
    Does not return any errors.

    """

    # Make sure it's a 2D array
    if len(A_in.shape) != 2:
        raise RuntimeError("A not a 2D array.")

    # Make sure it's square
    n = A_in.shape[0]
    if n != A_in.shape[1]:
        raise RuntimeError("A not square.")

    # Test for symmetry
    for i in range(n):
        for j in range(i):
            if A_in[i, j] != A_in[j, i]:
                raise RuntimeError("A not symmetric.")

    A = np.copy(A_in)
    p = np.arange(n)

    for k in range(n - 1):
        mu = k
        m_val = abs(A[k, k])
        # Find pivot.
        for j in range(k + 1, n):
            if abs(A[j, k]) > m_val:
                mu = j
                m_val = abs(A[j, k])

        # Pivot
        for j in range(n):
            temp = A[k, j]
            A[k, j] = A[mu, j]
            A[mu, j] = temp

        i = p[k]
        p[k] = p[mu]
        p[mu] = i

        if m_val > constants.float_eps:
            # If we have a non-zero pivot:
            for i in range(k + 1, n):
                A[i, k] = A[i, k] / A[k, k]

            for i in range(k + 1, n):
                for j in range(k + 1, n):
                    A[i, j] = A[i, j] - A[i, k] * A[k, j]

    return A, p


@jit(nopython=True)
def lup_solve(LU, p, b):

    """

    Performs LUP solve of square matrix A which has already undergone LUP
    decomposition (see lup_decomposition).

    This follows from Golub and van Loan (3.4.9)

    """

    # Make sure it's a 2D array.
    if len(LU.shape) != 2:
        raise RuntimeError("LU not a 2D array.")

    # Make sure it's square.
    n = LU.shape[0]
    if n != LU.shape[1]:
        raise RuntimeError("LU not square.")

    # Ensure dimension of matrix system agrees
    if n != b.shape[0] != p.shape[0]:
        raise RuntimeError("Matrix dimensions must agree.")

    # Make sure p is a permutation vector.
    if np.all(np.sort(p) != np.arange(p.shape[0])):
        raise RuntimeError("p not permutation vector.")

    # Get lower triangle.
    L = np.zeros((n, n))
    for i in range(n):
        L[i, i] = 1.0
        for j in range(i):
            L[i, j] = LU[i, j]

    # Get Upper Triangle
    U = np.zeros((n, n))
    for j in range(n):
        for i in range(0, j + 1):
            U[i, j] = LU[i, j]

    # Permute b vector
    b_perm = np.zeros(n)
    for i in range(n):
        b_perm[i] = b[p[i]]

    # Solve Ly = b_perm, storing y in x.
    x = lower_tri_solve(L, b_perm)

    # Solve Ux = y by back substitution.
    x = upper_tri_solve(U, x)

    return x


@jit("void(double[:, ::1], double, double)", nopython=True)
def _check_symmetry(A, atol, rtol):
    """Check to make sure a matrix is symmetric, 2D."""
    # Make sure it's a 2D array
    if len(A.shape) != 2:
        raise RuntimeError("A not a 2D array.")

    # Make sure it's square
    n = A.shape[0]
    if n != A.shape[1]:
        raise RuntimeError("A not square.")

    # Test for symmetry
    for i in range(n):
        for j in range(i):
            if not np.abs(A[i, j] - A[j, i]) < atol + rtol * abs(A[i, j]):
                raise RuntimeError("A not symmetric.")


@jit(nopython=True)
def nullspace_qr(A, tol):
    """
    Use QR factorization to compute a basis for the approximate
    null space of matrix A.

    Parameters
    ----------
    A : ndarray, shape (m, n) with m <= n
        The matrix whose null space basis to compute.
    tol : float, default = 1e-12
        Values below tol are considered zero.

    Returns
    -------
    output : ndarray, shape(k, n) where k is the rank of the null space
        A matrix whose columns span the null space of A.

        Every element in np.dot(A, output.T) should be approximately 0

    Notes
    -----
    .. QR factorization is not the most numerically stable technique,
       but it is very fast, and the stoichiometric matrices encountered
       are typically well-behaved.
    """
    _, n = A.shape
    q, r = np.linalg.qr(A.transpose(), "complete")
    rank = np.sum(np.abs(np.diag(r)) > tol)
    return q[:, n - rank - 1 :]
