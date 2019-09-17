import numpy as np

from . import solvers
from . import testcases


def step_diagnostics(n_random_test_cases=200, max_particles=4, max_compound_size=5):
    # Generate list of random test cases
    random_test_cases = []
    for i in range(n_random_test_cases):
        n_parts = np.random.randint(1, max_particles)
        max_cmp_size = np.random.randint(1, max_compound_size + 1)
        random_test_cases.append(
            testcases.random_elemental_test_case(n_parts, max_cmp_size)
        )

    step_tallies = np.empty((n_random_test_cases, 6))
    converged = np.empty(n_random_test_cases)
    n_trials = np.empty(n_random_test_cases)
    for tc in random_test_cases:
        conserv_vector = np.dot(tc["A"], tc["c0"])
        logx, converged_, n_trial, step_tally = solvers._solve_trust_region(
            tc["A"],
            tc["G"],
            conserv_vector,
            max_iters=1000,
            tol=0.0000001,
            delta_bar=1000.0,
            eta=0.125,
            min_delta=1.0e-12,
            max_trials=100,
            perturb_scale=100.0,
        )
        step_tallies[i] = step_tally
        converged[i] = converged_
        n_trials[i] = n_trial

    return step_tallies, converged, n_trials
