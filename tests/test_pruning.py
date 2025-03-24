import numpy as np
from numba import boolean
import eqtk


def test_prune_NK():
    N = np.array(
        [
            [-1, 1, 0, 0, 0, 0],
            [-1, 0, -1, 1, 0, 0],
            [0, -2, 0, 0, 1, 0],
            [0, -1, -1, 0, 0, 1],
        ],
        dtype=float,
    )
    minus_log_K = np.array([1, 2, 3, 4], dtype=float)

    # All present
    x0 = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.solvers._prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Also all present
    x0 = np.array([1, 0, 3, 0, 0, 0], dtype=float)
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.solvers._prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Also all present
    x0 = np.array([0, 0, 0, 0, 0, 6], dtype=float)
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.solvers._prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Only entries 0, 1 and 4 present
    x0 = np.array([0, 2, 0, 0, 0, 0], dtype=float)
    x0_target = np.array([0, 2, 0], dtype=float)
    N_target = np.array([[-1, 1, 0], [0, -2, 1]], dtype=float)
    minus_log_K_target = np.array([1, 3], dtype=float)
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.solvers._prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N_target)
    assert np.array_equal(minus_log_K_new, minus_log_K_target)
    assert np.array_equal(x0_new, x0_target)

    # Only entry 2 present
    x0 = np.array([0, 0, 3, 0, 0, 0], dtype=float)
    N_target = np.array([[]])
    minus_log_K_target = np.array([])
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.solvers._prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N_target)
    assert np.array_equal(minus_log_K_new, minus_log_K_target)
    assert np.array_equal(x0_new, x0)

    N = np.array(
        [
            [-1, 0, 1, 0, 0, 0],
            [-1, -1, 0, 1, 0, 0],
            [0, -2, 0, 0, 1, 0],
            [0, -1, -1, 0, 0, 1],
        ],
        dtype=float,
    )
    minus_log_K = np.array([1, 2, 3, 4], dtype=float)

    # All present
    x0 = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.solvers._prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Also all present
    x0 = np.array([1, 2, 0, 0, 0, 0], dtype=float)
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.solvers._prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Also all present
    x0 = np.array([0, 0, 0, 0, 0, 6], dtype=float)
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.solvers._prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N)
    assert np.array_equal(minus_log_K_new, minus_log_K)
    assert np.array_equal(x0_new, x0)

    # Only entries 0 and 2 present
    x0 = np.array([1, 0, 3, 0, 0, 0], dtype=float)
    x0_target = np.array([1, 3], dtype=float)
    N_target = np.array([[-1, 1]], dtype=float)
    minus_log_K_target = np.array([1], dtype=float)
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.solvers._prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N_target)
    assert np.array_equal(minus_log_K_new, minus_log_K_target)
    assert np.array_equal(x0_new, x0_target)

    # Only entries 1 and 4 present
    x0 = np.array([0, 2, 0, 0, 0, 0], dtype=float)
    x0_target = np.array([2, 0], dtype=float)
    N_target = np.array([[-2, 1]], dtype=float)
    minus_log_K_target = np.array([3], dtype=float)
    N_new, minus_log_K_new, x0_new, _, _ = eqtk.solvers._prune_NK(N, minus_log_K, x0)
    assert np.array_equal(N_new, N_target)
    assert np.array_equal(minus_log_K_new, minus_log_K_target)
    assert np.array_equal(x0_new, x0_target)

    # Reactions with solvent dissociation
    N = np.array(
        [[1, 0, 1, 0, -1, 0], [1, 0, 0, 1, 0, -1], [1, 1, 1, 0, 0, 0]], dtype=float
    )
    minus_log_K = np.array([1, 2, 3], dtype=float)

    # No pruning
    for x0_val in [
        [1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0],
        [1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1],
    ]:
        x0 = np.array(x0_val, dtype=float)
        N_new, minus_log_K_new, x0_new, _, _ = eqtk.solvers._prune_NK(
            N, minus_log_K, x0
        )
        assert np.array_equal(N_new, N)
        assert np.array_equal(minus_log_K_new, minus_log_K)
        assert np.array_equal(x0_new, x0)

    # Only entries 0, 1, 2, and 4
    x0 = np.zeros(6, dtype=float)
    x0_target = np.zeros(4, dtype=float)
    N_target = np.array([[1, 0, 1, -1], [1, 1, 1, 0]], dtype=float)
    minus_log_K_target = np.array([1.0, 3.0])
    active_compounds_target = np.array([True, True, True, False, True, False])
    (
        N_new,
        minus_log_K_new,
        x0_new,
        active_compounds,
        active_reactions,
    ) = eqtk.solvers._prune_NK(N, minus_log_K, x0)
    assert np.array_equal(active_compounds, active_compounds_target)
    assert np.array_equal(N_new, N_target)
    assert np.array_equal(minus_log_K_new, minus_log_K_target)
    assert np.array_equal(x0_new, x0_target)


def test_prune_AG():
    A = np.array([[1, 0, 1, 1, 0, 1], [0, 1, 0, 1, 2, 1]], dtype=float)
    G = np.array([1, 2, 3, 4, 5, 6], dtype=float)

    # No pruning
    for x0_val in [[1, 2, 3, 4, 5, 6], [1, 2, 0, 0, 0, 0], [0, 0, 0, 0, 0, 6]]:
        x0 = np.array(x0_val, dtype=float)
        A_new, G_new, x0_new, active_compounds = eqtk.solvers._prune_AG(A, G, x0)
        assert np.array_equal(active_compounds, np.ones(6, dtype=np.bool_))
        assert np.array_equal(A_new, A)
        assert np.array_equal(G_new, G)
        assert np.array_equal(x0_new, x0)

    # Only species 1 and 4
    x0 = np.array([0, 0, 0, 0, 5, 0], dtype=float)
    x0_prune = np.array([0, 5], dtype=float)
    A_target = np.array([[1, 2]], dtype=float)
    G_target = np.array([2, 5], dtype=float)
    A_new, G_new, x0_new, active_compounds = eqtk.solvers._prune_AG(A, G, x0)
    assert np.array_equal(
        active_compounds, np.array([0, 1, 0, 0, 1, 0], dtype=np.bool_)
    )
    assert np.array_equal(A_new, A_target)
    assert np.array_equal(G_new, G_target)
    assert np.array_equal(x0_new, x0_prune)

    # Simple case, binary binding
    A = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    G = np.array([0, 0, 1], dtype=float)

    # No pruning
    for x0_val in [[1, 1, 1], [1, 1, 0], [0, 0, 1]]:
        x0 = np.array(x0_val, dtype=float)
        A_new, G_new, x0_new, active_compounds = eqtk.solvers._prune_AG(A, G, x0)
        assert np.array_equal(active_compounds, np.ones(3, dtype=np.bool_))
        assert np.array_equal(A_new, A)
        assert np.array_equal(G_new, G)
        assert np.array_equal(x0_new, x0)

    # Only keep element 0
    x0 = np.array([1, 0, 0], dtype=float)
    x0_prune = np.array([1.0])
    A_target = np.array([[1]], dtype=float)
    G_target = np.array([0.0])
    A_new, G_new, x0_new, active_compounds = eqtk.solvers._prune_AG(A, G, x0)
    assert np.array_equal(active_compounds, np.array([1, 0, 0], dtype=np.bool_))
    assert np.array_equal(A_new, A_target)
    assert np.array_equal(G_new, G_target)
    assert np.array_equal(x0_new, x0_prune)

    # Only keep element 1
    x0 = np.array([0, 1, 0], dtype=float)
    x0_prune = np.array([1.0])
    A_target = np.array([[1]], dtype=float)
    G_target = np.array([0.0])
    A_new, G_new, x0_new, active_compounds = eqtk.solvers._prune_AG(A, G, x0)
    assert np.array_equal(active_compounds, np.array([0, 1, 0], dtype=np.bool_))
    assert np.array_equal(A_new, A_target)
    assert np.array_equal(G_new, G_target)
    assert np.array_equal(x0_new, x0_prune)

    # A trickier case
    A = np.array(
        [[1, 0, 1, 1, 1], [0, 1, 2, 1, 0], [1, 1, 0, 1, 0], [0, 0, 0, 0, 1]],
        dtype=float,
    )
    G = np.array([1, 2, 3, 4, 5], dtype=float)

    # No pruning
    for x0_val in [[1, 1, 0, 0, 1], [0, 0, 0, 1, 1], [0, 1, 0, 0, 1], [1, 1, 1, 1, 1]]:
        x0 = np.array(x0_val, dtype=float)
        A_new, G_new, x0_new, active_compounds = eqtk.solvers._prune_AG(A, G, x0)
        assert np.array_equal(active_compounds, np.ones(5, dtype=np.bool_))
        assert np.array_equal(A_new, A)
        assert np.array_equal(G_new, G)
        assert np.array_equal(x0_new, x0)

    # Only entry 2
    x0 = np.array([0, 0, 1, 0, 0], dtype=float)
    x0_prune = np.array([1], dtype=float)
    A_target = np.array([[1], [2]], dtype=float)
    G_target = np.array([3], dtype=float)
    A_new, G_new, x0_new, active_compounds = eqtk.solvers._prune_AG(A, G, x0)
    assert np.array_equal(active_compounds, np.array([0, 0, 1, 0, 0], dtype=np.bool_))
    assert np.array_equal(A_new, A_target)
    assert np.array_equal(G_new, G_target)
    assert np.array_equal(x0_new, x0_prune)

    # All but last entry
    A_target = A[:-1, :-1]
    G_target = np.array([1, 2, 3, 4], dtype=float)
    for x0_val in [[1, 1, 0, 0, 0], [0, 0, 0, 1, 0], [1, 1, 1, 1, 0]]:
        x0 = np.array(x0_val, dtype=float)
        x0_prune = x0[:-1]
        A_new, G_new, x0_new, active_compounds = eqtk.solvers._prune_AG(A, G, x0)
        assert np.array_equal(
            active_compounds, np.array([1, 1, 1, 1, 0], dtype=np.bool_)
        )
        assert np.array_equal(A_new, A_target)
        assert np.array_equal(G_new, G_target)
        assert np.array_equal(x0_new, x0_prune)
