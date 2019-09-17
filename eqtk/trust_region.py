"""
Trust region method of optimization of convex objective function.
"""
import numpy as np

from . import constants
from . import linalg
from . import numba_check

have_numba, jit = numba_check.numba_check()


@jit("double[::1](double[::1], double[::1], double[:, ::1])", nopython=True)
def compute_logx(mu, G, A):
    return -G + np.dot(mu, A)


@jit("double[::1](double[::1], double)", nopython=True)
def rescale_x(x, log_scale_param):
    """Rescale the concentrations.
    """
    return x * np.exp(-log_scale_param)


@jit("double[::1](double[::1], double)", nopython=True)
def rescale_constraint_vector(constraint_vector, log_scale_param):
    """Rescale the constraint vector by the maximal concentration.
    """
    return constraint_vector * np.exp(-log_scale_param)


@jit(
    "double(double[::1], double[::1], double[::1], double[:, ::1], double[::1])",
    nopython=True,
)
def obj_fun(mu, x, G, A, constraint_vector):
    """
    Compute the negative dual function for equilibrium calcs.

    The constraint vector must be previously rescaled appropriately to
    match the `log_scale_param`.
    """
    return np.sum(x) - np.dot(mu, constraint_vector)


@jit("double[::1](double[::1], double[:, ::1], double[::1])", nopython=True)
def grad(x, A, constraint_vector):
    """
    Computes the gradient of the negative dual function.

    The constraint vector must be previously rescaled appropriately to
    match the `log_scale_param`.
    """
    return -constraint_vector + np.dot(A, x)


@jit("double[:, ::1](double[::1], double[:, ::1])", nopython=True)
def hes(x, A):
    """
    Computes the gradient of the negative dual function.
    """
    m = A.shape[0]
    B = np.empty((m, m))
    for i in range(m):
        for j in range(i + 1):
            B[j][i] = np.dot(x, A[j, :] * A[i, :])
            B[i][j] = B[j][i]
    return B


@jit("double[::1](double[:, ::1])", nopython=True)
def inv_scaling(B):
    """Computes diagonal of scaling matrix."""
    return 1.0 / np.sqrt(np.diag(B))


@jit("double[::1](double[::1], double[::1])", nopython=True)
def scaled_grad(g, inv_d):
    """Computes the scaled gradient."""
    return inv_d * g


@jit("double[:, ::1](double[:, ::1], double[::1])", nopython=True)
def scaled_hes(B, inv_d):
    """Computes the scaled Hessian."""
    return linalg.diag_multiply(B, inv_d)


@jit("Tuple((double[::1], int64))(double[::1], double[:, ::1], double)", nopython=True)
def search_direction_dogleg(g, B, delta):
    """
    Computes the search direction using the dogleg method (Nocedal and Wright,
    page 71).  Notation is consistent with that in this reference.

    p = direction to step
    g = gradient
    B = Hessian
    delta = radius of trust region

    Due to the construction of the problem, the minimization routine
    to find tau can be solved exactly by solving a quadratic.  There
    can be precision issues with this, so this is checked.  If the
    argument of the square root in the quadratic formula is negative,
    there must be a precision error, as such a situation is not
    possible in the construction of the problem.

    Returns:
      p, the search direction and also an integer,

      1 if step was a pure Newton step (didn't hit trust region boundary)
      2 if the step was purely Cauchy in nature (hit trust region boundary)
      3 if the step was a dogleg step (part Newton and part Cauchy)
      4 if Cholesky decomposition failed and we had to take a Cauchy step
      5 if Cholesky decompostion failed but we would've taken Cauchy step anyway
      6 if the dogleg calculation failed (should never happen)
    """
    # Useful to have around
    delta2 = delta ** 2

    # Rescale
    inv_d = inv_scaling(B)
    g = scaled_grad(g, inv_d)
    B = scaled_hes(B, inv_d)

    # Compute Newton step, pB
    neg_pB, cholesky_success = linalg.solve_pos_def(B, g)
    pB = -neg_pB

    # If Newton step is in trust region, take it
    if cholesky_success:
        pB2 = np.dot(pB, pB)
        if pB2 <= delta2:
            return inv_d * pB, 1

    # Compute Cauchy step, pU
    pU = -np.dot(g, g) / np.dot(g, np.dot(B, g)) * g
    pU2 = np.dot(pU, pU)
    if pU2 >= delta2:  # In this case, just take Cauchy step, 0 < tau <= 1
        tau = np.sqrt(delta2 / pU2)
        p = tau * pU
        if not cholesky_success:
            return inv_d * p, 5  # Cholesky failure, but doesn't matter, just use Cauchy
        else:
            return inv_d * p, 2  # Took Cauchy step

    if not cholesky_success:  # Failed computing Newton, have to take Cauchy
        return inv_d * pU, 4

    # Compute the dogleg step
    pBpU = np.dot(pB, pU)  # Needed for dogleg computation

    # Compute constants for quadratic formula for solving
    # ||pU + beta (pB-pU)||^2 = delta2, with beta = tau - 1
    a = pB2 + pU2 - 2.0 * pBpU  # a > 0
    b = 2.0 * (pBpU - pU2)  # b can be positive or negative
    c = pU2 - delta2  # c < 0 since pU2 < delta 2 to get here
    q = -0.5 * (b + np.sign(b) * np.sqrt(b ** 2 - 4.0 * a * c))

    # Choose correct (positive) root (don't have to worry about a = 0 because
    # if pU \approx pB, we would have already taken Newton step)
    if abs(b) < constants._float_eps:
        beta = np.sqrt(-c / a)
    elif b < 0.0:
        beta = q / a
    else:  # b > 0
        beta = c / q

    # Take the dogleg step
    if 0.0 <= beta <= 1.0:  # Should always be the case
        p = pU + beta * (pB - pU)
        return inv_d * p, 3
    else:  # Something is messed up, take Cauchy step. Only get here if
        # precision issues with beta
        return inv_d * pU, 6


@jit("boolean(double[::1], double[::1], double)", nopython=True)
def check_tol(g, tol, log_scale_param):
    """
    Check to see if the tolerance has been met.  Returns False if
    tolerance is met.
    """
    s = np.exp(log_scale_param)
    for tolerance, gradient in zip(tol, g):
        if abs(gradient) > tolerance / s:
            return True
    return False


@jit(
    "Tuple((double[::1], boolean, int64[::1]))(double[::1], double[::1], double[:, ::1], double[::1], double[::1], double, double, double, double)",
    nopython=True,
)
def trust_region_convex_unconstrained(
    mu0,
    G,
    A,
    constraint_vector,
    tol,
    max_iters=10000,
    delta_bar=1000.0,
    eta=0.125,
    min_delta=1e-12,
):
    """
    Performs a unconstrained trust region optimization on a convex
    function where the gradient and hessian have analytical forms and
    the functions to calculate them are given.  This algorithm is
    given in Nocedal and Wright, Numerical Optimization, 1999, page
    68, with the dogleg method on page 71.

    mu0: initial guess of optimal mu
    f: the objective function of the form f(mu, *params)
    grad: Function to compute the gradient of f.  grad(mu, *params)
    hes: Function to compute the gradient of f.  hes(mu, *params)

    In the objective function is strictly convex, the trust region
    algorithm is globally convergent.  However, on occasion, when
    close to the optimum, the gradient may be close to zero, but not
    within tolerance, but the value of the objective function is the
    optimal value to within machine precision.  This results in the
    criterion for shrinking the trust region to be triggered (rho ≈ 0),
    and the trust region shrinks to a infinitum.  As a remedy for this, we
    specify min_delta to be the minimal trust region radius allowed.  When
    the trust region radius just below
    this value, we are very very close to the minimum.  The algorithm
    then proceeds with Newton steps until convergence is achieved or
    the Newton step fails to decrease the norm of the gradient.

    step_tally enties:

      0 if step was a pure Newton step (didn't hit trust region boundary)
      1 if the step was purely Cauchy in nature (hit trust region boundary)
      2 if the step was a dogleg step (part Newton and part Cauchy)
      3 if Cholesky decomposition failed and we had to take a Cauchy step
      4 if Cholesky decompostion failed but we would've taken Cauchy step anyway
      5 if the dogleg calculation failed (should never happen)
      6 if step was a last ditch Newton step

    Suggested defaults:
    max_iters=10000,
    delta_bar=1000.0,
    eta=0.125,
    min_delta=1e-12,
    """
    # Initializations
    delta = 0.99 * delta_bar
    iters = 0
    n_no_step = 0
    mu = np.copy(mu0)
    step_tally = np.zeros(7, dtype=np.int64)

    # Calculate the concentrations
    logx = compute_logx(mu, G, A)
    log_scale_param = logx.max()
    x = np.exp(logx - log_scale_param)
    constr_vec = rescale_constraint_vector(constraint_vector, log_scale_param)

    # Calculate the function, gradient, hessian, and scaling matrix
    f = obj_fun(mu, x, G, A, constr_vec)
    g = grad(x, A, constr_vec)
    B = hes(x, A)

    # Run the trust region
    while (
        iters < max_iters and check_tol(g, tol, log_scale_param) and delta >= min_delta
    ):
        # Solve for search direction
        p, search_result = search_direction_dogleg(g, B, delta)
        step_tally[search_result - 1] += 1

        # New candidate mu, x
        mu_new = mu + p
        logx = compute_logx(mu_new, G, A)
        x_new = np.exp(logx - log_scale_param)

        # Calculate rho, ratio of actual to predicted reduction
        f_new = obj_fun(mu_new, x_new, G, A, constr_vec)
        rho = (f_new - f) / (np.dot(g, p) + np.dot(p, np.dot(B, p)) / 2.0)

        # Adjust delta
        if rho < 0.25:
            delta /= 4.0
        elif rho > 0.75 and abs(np.linalg.norm(p) - delta) < constants._float_eps:
            delta = min(2.0 * delta, delta_bar)

        # Make step based on rho
        if rho > eta:
            log_scale_param_new = logx.max()
            rescale_factor = np.exp(log_scale_param - log_scale_param_new)
            log_scale_param = log_scale_param_new
            mu = mu_new
            x = x_new * rescale_factor
            f = f_new * rescale_factor
            constr_vec = rescale_constraint_vector(constraint_vector, log_scale_param)
            g = grad(x, A, constr_vec)
            B = hes(x, A)

        iters += 1

    # Try Newton stepping to the solution if trust region failed (should not get here)
    if delta <= min_delta:
        # If we get here, it is because of numerical precision issues in computing
        # f_new - f. When this is effectively zero, we cannot compute rho,
        # so the trust region keeps shrinking. This also means we are very
        # close to the minimum, so we will attempt to continue with Newton
        # stepping.
        newton_success = True
        while (
            newton_success and check_tol(g, tol, log_scale_param) and iters < max_iters
        ):
            neg_pB, newton_success = linalg.solve_pos_def(B, g)

            if newton_success:
                pB = -neg_pB
                mu_new = mu + pB
                logx = compute_logx(mu_new, G, A)
                log_scale_param_new = logx.max()
                x_new = np.exp(logx - log_scale_param_new)
                constr_vec_new = rescale_constraint_vector(
                    constraint_vector, log_scale_param_new
                )
                g_new = grad(x_new, A, constr_vec)

                # If decreased the gradient, take the step.
                rescale_factor = np.exp(log_scale_param - log_scale_param_new)
                if np.linalg.norm(g_new) < np.linalg.norm(g) / rescale_factor:
                    mu = mu_new
                    x = x_new
                    log_scale_param = log_scale_param_new
                    constr_vec = constr_vec_new
                    g = g_new
                    B = hes(x_new, A)

                    step_tally[6] += 1
                else:
                    newton_success = False
            iters += 1

        converged = newton_success and iters < max_iters
    else:
        converged = iters < max_iters

    return mu, converged, step_tally
