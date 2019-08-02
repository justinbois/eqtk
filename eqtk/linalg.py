import sys
import numpy as np

from . import constants
from . import numba_check


def diag_multiply(A, d):
    """Compute the D . A . D, where d is the diagonal of diagonal
    matrix D.
    """
    B = np.empty_like(A)

    for j in range(len(d)):
        B[:, j] = d[j] * A[:, j]

    for i in range(len(d)):
        B[i, :] = d[i] * B[i, :]

    return B


def solve_pos_def(A, b):
    """Solve A x = b, where A is known to be positive definite.

    Parameters
    ----------
    A : ndarray, shape (n, n)
        Symmetric, real, positive definite or positive semidefinite
        matrix.
    b : ndarray, shape (n,)
        Array of real numbers.

    Returns
    -------
    output : ndarray, shape (n,)
        Solution x to A x = b.
    success : bool
        False if Cholesky decomposition failed.
    """
    L, p, success = modified_cholesky(A)

    if not success:
        return np.zeros(len(b)), success

    return modified_cholesky_solve(L, p, b), success


def _check_symmetry(A, atol=1e-8, rtol=1e-5):
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


def modified_cholesky(A):
    """
    Modified Cholesky decomposition.
    Performs modified Cholesky decomposition based on the algorithm
    GMW81 in Fang, O'Leary, 2006. Modified Cholesky Algorithms: A Catalog
    with New Approaches. From the matrix A

    Parameters
    ----------
    A : ndarray
        Symmetric, real, positive definite or positive semidefinite
        matrix.

    Returns
    -------
    L : ndarray, shape A.shape
        Lower triangulatar matrix with Cholesky decomposition
    p : ndarray, shape A.shape[0]
        Permutation vector defining which columns of permutation matrix
        have ones.
    success : bool
        True if Cholesky decomposition was successful and False otherwise,
        usually due to matrix not being positive semidefinite.
    """
    _check_symmetry(A)
    n = A.shape[0]

    L = np.copy(A)
    p = np.arange(n)

    # Keep track if factorization was successful.
    success = True

    xi = 0
    for i in range(n):
        for j in range(i):
            temp = abs(L[i, j])
            xi = max(xi, temp)

    eta = 0
    for i in range(n):
        temp = abs(L[i, i])
        eta = max(eta, temp)

    if n > 1:
        beta = np.sqrt(max(eta, xi / np.sqrt(n * n - 1)))
    else:
        beta = np.sqrt(eta)
    beta = max(beta, constants.float_eps)

    for k in range(n):
        # Pick a pivot.
        mu_val = L[k, k]
        mu = k
        for i in range(k + 1, n):
            temp = L[i, i]
            if mu_val < temp:
                mu = i
                mu_val = temp

        # Diagonal pivot k <=> mu
        i = p[mu]
        p[mu] = p[k]
        p[k] = i

        for i in range(k):
            temp = L[k, i]
            L[k, i] = L[mu, i]
            L[mu, i] = temp

        temp = L[k, k]
        L[k, k] = L[mu, mu]
        L[mu, mu] = temp
        for i in range(k + 1, mu):
            temp = L[i, k]
            L[i, k] = L[mu, i]
            L[mu, i] = temp

        for i in range(mu + 1, n):
            temp = L[i, k]
            L[i, k] = L[i, mu]
            L[i, mu] = temp

        # Compute c_sum
        c_sum = 0
        for i in range(k + 1, n):
            c_sum = max(c_sum, abs(L[i, k]))
        c_sum /= beta
        c_sum = c_sum * c_sum

        # Make sure L is semi-positive definite.
        if L[k, k] < 0:
            # Otherwise the factorization was not completed successfully.
            success = False

        temp = abs(L[k, k])
        temp = max(temp, constants.float_eps * eta)
        temp = max(temp, c_sum)
        L[k, k] = np.sqrt(temp)

        # Compute the current column of L.
        for i in range(k + 1, n):
            L[i, k] /= L[k, k]

        # Adjust the \bar{L}
        for j in range(k + 1, n):
            for i in range(j, n):
                L[i, j] -= L[i, k] * L[j, k]

        # Just keep lower triangle
        for i in range(0, n - 1):
            L[i, i + 1 :] = 0.0

    return L, p, success


def modified_cholesky_solve(L, p, b):
    """
    Solve system Ax = b, with P A P^T = L L^T post-Cholesky decomposition.
    Parameters
    ----------
    L : ndarray
        Square lower triangulatar matrix from Cholesky decomposition
    p : ndarray, shape L.shape[0]
        Permutation vector defining which columns of permutation matrix
        have ones.
    b : ndarray, shape L.shape[0]
        Right hand side of Ax = b equation being solved.
    Returns
    -------
    x : ndarray, shape L.shape[0]
        Solution to Ax = b.
    """
    # Make sure it's a 2D array.
    if len(L.shape) != 2:
        raise RuntimeError("L not a 2D array.")

    # Make sure it's square.
    n = L.shape[0]
    if n != L.shape[1]:
        raise RuntimeError("L not square.")

    # Test for lower triangular.
    for j in range(n):
        for i in range(0, j):
            if L[i, j] != 0:
                raise RuntimeError("L is not lower triangular.")

    # Ensure dimension of matrix system agrees
    if n != b.shape[0] != p.shape[0]:
        raise RuntimeError("Matrix dimensions must agree.")

    # Make sure p is a permutation vector.
    if not np.all(np.sort(p) == np.arange(p.shape[0])):
        raise RuntimeError("p not permutation vector.")

    U = L.transpose()
    xp = np.zeros(n)
    for i in range(n):
        xp[i] = b[p[i]]

    # Solve Ly = b storing y in xp.
    x = lower_tri_solve(L, xp)

    # Solve L^T x = y by back substitution.
    xp = upper_tri_solve(U, x)

    for i in range(n):
        x[p[i]] = xp[i]

    return x


def lower_tri_solve(L, b):
    """
    Solves the lower triangular system Lx = b.
    Uses column-based forward substitution, outlined in algorithm
    3.1.3 of Golub and van Loan.
    Parameters
    ----------
    L : ndarray
        Square lower triangulatar matrix (including diagonal)
    b : ndarray, shape L.shape[0]
        Right hand side of Lx = b equation being solved.
    Returns
    -------
    x : ndarray
        Solution to Lx = b.
    """

    # Make sure it's a 2D array.
    if len(L.shape) != 2:
        raise RuntimeError("L not a 2D array.")

    # Make sure it's square.
    n = L.shape[0]
    if n != L.shape[1]:
        raise RuntimeError("L not square.")

    # Test for lower triangular.
    for j in range(n):
        for i in range(0, j):
            if L[i, j] != 0:
                raise RuntimeError("L is not lower triangular.")

    # Ensure dimension of matrix system agrees
    if n != b.shape[0]:
        raise RuntimeError("Matrix dimensions must agree.")

    # Solve Lx = b.
    x = np.copy(b)
    for j in range(n - 1):
        if abs(L[j, j]) > constants.float_eps:
            x[j] /= L[j, j]
            for i in range(j + 1, n):
                x[i] -= x[j] * L[i, j]
        else:
            x[j] = 0.0

    if n > 0:
        if abs(L[n - 1, n - 1]) > constants.float_eps:
            x[n - 1] /= L[n - 1, n - 1]
        else:
            x[n - 1] = 0.0

    return x


def upper_tri_solve(U, b):
    """
    Solves the lower triangular system Ux = b.
    Uses column-based forward substitution, outlined in algorithm
    3.1.4 of Golub and van Loan.
    Parameters
    ----------
    U: ndarray
        Square upper triangulatar matrix (including diagonal)
    b : ndarray, shape L.shape[0]
        Right hand side of Ux = b equation being solved.
    Returns
    -------
    x : ndarray
        Solution to Ux = b.
    """

    # Make sure it's a 2D array.
    if len(U.shape) != 2:
        raise RuntimeError("U not a 2D array.")

    # Make sure it's square.
    n = U.shape[0]
    if n != U.shape[1]:
        raise RuntimeError("U not square.")

    # Test for upper triangular.
    for i in range(n):
        for j in range(0, i):
            if U[i, j] != 0:
                raise RuntimeError("U is not upper triangular.")

    # Ensure dimension of matrix system agrees
    if n != b.shape[0]:
        raise RuntimeError("Matrix dimensions must agree.")

    # Solve Ux = b by back substitution.
    x = np.copy(b)
    for j in range(n - 1, 0, -1):
        if abs(U[j, j]) > constants.float_eps:
            x[j] /= U[j, j]
            for i in range(0, j):
                x[i] -= x[j] * U[i, j]
        else:
            x[j] = 0.0

    if n > 0:
        if abs(U[0, 0]) > constants.float_eps:
            x[0] /= U[0, 0]
        else:
            x[0] = 0.0

    return x


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


def nullspace_svd(A, tol=1e-12):
    """
    Calculate a basis for the approximate null space of A, consider any
    eigenvalue less than tolerance as 0

    Parameters
    ----------
    A : ndarray
        A matrix of shape (M, N) M <= N
    tol :
        The floor in absolute value for eigenvalues to be considered
        non-zero

    Returns
    -------
    N : ndarray
        A matrix of shape (K, N) where K is the rank of the null space,
        at least N-M

        Every element in np.dot(A, N.T) should be approximately 0
    """
    u, sigma, v_trans = np.linalg.svd(A)
    nonzero_inds = (sigma >= tol).sum()
    return v_trans[nonzero_inds:, :].transpose()


def nullspace_qr(A, tol=1e-12):
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


if numba_check.numba_check():
    import numba

    _check_symmetry = numba.jit(_check_symmetry, nopython=True)
    diag_multiply = numba.jit(diag_multiply, nopython=True)
    modified_cholesky = numba.jit(modified_cholesky, nopython=True)
    modified_cholesky_solve = numba.jit(modified_cholesky_solve, nopython=True)
    lower_tri_solve = numba.jit(lower_tri_solve, nopython=True)
    upper_tri_solve = numba.jit(upper_tri_solve, nopython=True)
    lup_decomposition = numba.jit(lup_decomposition, nopython=True)
    lup_solve = numba.jit(lup_solve, nopython=True)
    nullspace_svd = numba.jit(nullspace_svd, nopython=True)
    solve_pos_def = numba.jit(solve_pos_def, nopython=True)
else:
    solve_pos_def = np.linalg.solve
