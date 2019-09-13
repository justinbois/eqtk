import numpy as np
import pandas as pd
import numba


@numba.jit(nopython=True)
def _adjust_sigma(acc_rate, sigma):
    """
    Tune sigma in proposal distribution.

    Parameters
    ----------
    acc_rate : float
        The acceptance rate.
    sigma : ndarray
        Array of standard deviations for Gaussian proposal
        distribution.

    Returns
    -------
    output : ndarray
        Updated `sigma` values.
    """
    if acc_rate < 0.001:
        return sigma * 0.1
    elif acc_rate < 0.05:
        return sigma * 0.5
    elif acc_rate < 0.2:
        return sigma * 0.9
    elif acc_rate > 0.95:
        return sigma * 10.0
    elif acc_rate > 0.75:
        return sigma * 2.0
    elif acc_rate > 0.5:
        return sigma * 1.1
    else:
        return sigma


def mh_sample(logpost, x0, sigma=None, discrete=None, args=(), n_burn=1000,
              n_steps=1000, tune_interval=100, variable_names=None,
              return_acceptance_rate=False):
    """
    Parameters
    ----------
    logpost : function
        The function to compute the log posterior. It has call
        signature `logpost(x, *args)`, and must by compiled with
        numba in nopython mode.
    x0 : ndarray, shape (n_variables,)
        The starting location of a walker in parameter space.
    sigma : ndarray, shape (n_variables,)
        The standard deviations for the proposal distribution.
        If None, takes all values to be one.
    discrete : ndarray of bools, shape (n_variables,)
        discrete[i] is True if variable i is discrete and False
        otherwise. If None (default), all entries are False.
    args : tuple
        Additional arguments passed to `logpost()` function.
    n_burn : int, default 1000
        Number of burn-in steps.
    n_steps : int, default 1000
        Number of steps to take after burn-in.
    tune_interval : int, default 100
        Number of steps to use when determining acceptance
        fraction for tuning.
    variable_names : list, length n_variables
        List of names of variables. If None, then variable names
        are sequential integers.
    return_acceptance_rate : bool, default False
        If True, also return acceptance rate.

    Returns
    -------
    output : DataFrame
        The first `n_variables` columns contain the samples.
        Additionally, column 'lnprob' has the log posterior value
        at each sample.
    """

    @numba.jit(nopython=True)
    def mh_step(x, logpost_current, sigma, discrete, args=()):
        """
        Parameters
        ----------
        x : ndarray, shape (n_variables,)
            The present location of the walker in parameter space.
        logpost_current : float
            The current value of the log posterior.
        sigma : ndarray, shape (n_variables, )
            The standard deviations for the proposal distribution.
        discrete : ndarray of bools, shape (n_variables,)
            discrete[i] is True if variable i is discrete and False
            otherwise. If None (default), all entries are False.
        args : tuple
            Additional arguments passed to `logpost()` function.

        Returns
        -------
        output : ndarray, shape (n_variables,)
            The position of the walker after the Metropolis-Hastings
            step. If no step is taken, returns the inputted `x`.
        """
        # Draw the next step
        x_next = np.empty(len(x), dtype=x.dtype)
        for i, x_val in enumerate(x):
            if discrete[i]:
                x_next[i] = np.round(np.random.normal(x[i], sigma[i]))
            else:
                x_next[i] = np.random.normal(x[i], sigma[i])

        # Compute log posterior
        logpost_new = logpost(x_next, *args)

        # Compute the log Metropolis ratio
        log_r = logpost_new - logpost_current

        # Accept or reject step
        if log_r >= 0 or np.random.random() < np.exp(log_r):
            return x_next, logpost_new, True
        else:
            return x, logpost_current, False

        # Never tune if asked not to
        if tune_interval <= 0:
            tune_interval = n_burn + n_steps
        elif tune_interval < 50:
            raise RuntimeError('Tune interval should be at least 50.')


    @numba.jit(nopython=True)
    def _metropolis_sample(x0, sigma, discrete, n_burn, n_steps,
                           tune_interval=100, args=()):
        """
        Numba'd sampler.
        """

        # Initialize
        x = np.copy(x0)
        n_accept = 0
        n_accept_total = 0
        n = 0
        n_tune_steps = 0
        n_continuous_steps = 0
        lnprob = np.empty(n_steps)
        logpost_current = logpost(x, *args)

        # Burn in
        while n < n_burn:
            while n_tune_steps < tune_interval and n < n_burn:
                x, logpost_current, accept = mh_step(
                            x, logpost_current, sigma, discrete, args)
                n += 1
                n_tune_steps += 1
                n_accept += accept
            sigma = _adjust_sigma(n_accept/tune_interval, sigma)
            n_accept = 0
            n_tune_steps = 0

        # Samples
        x_samples = np.empty((n_steps, len(x)))
        n = 0
        while n < n_steps:
            while n_tune_steps < tune_interval and n < n_steps:
                x, logpost_current, accept = mh_step(
                            x, logpost_current, sigma, discrete, args)
                x_samples[n,:] = x
                lnprob[n] = logpost_current
                n_tune_steps += 1
                n_accept += accept
                n_accept_total += accept
                n += 1
            sigma = _adjust_sigma(n_accept/tune_interval, sigma)
            n_accept = 0
            n_tune_steps = 0

        return x_samples, lnprob, n_accept_total/n_steps

    # Use default indices for variable names
    if variable_names is None:
        variable_names = list(range(len(x0)))

    # Set sigma to be default if not provided
    if sigma is None:
        sigma = np.ones(len(x0))

    # Default to continuous variables
    if discrete is None:
        discrete = np.array([False]*len(x0))
    else:
        discrete = np.array(discrete)

    # Grab the samples in a NumPy array
    x_samples, lnprob, acc_rate = _metropolis_sample(
            x0, sigma, discrete, n_burn, n_steps,
            tune_interval=tune_interval, args=args)

    df = pd.DataFrame(columns=variable_names, data=x_samples)
    df['lp__'] = lnprob

    if return_acceptance_rate:
        return df, acc_rate
    return df
