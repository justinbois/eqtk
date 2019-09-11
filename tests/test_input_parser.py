import numpy as np
import pytest

import eqtk.parsers


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
        eqtk.parsers.parse_input([], N, K, None, None, *tuple([None] * 5))
    excinfo.match(
        "Dimension mismatch between `c0` and inputted chemical species via `N`."
    )

    with pytest.raises(ValueError) as excinfo:
        eqtk.parsers.parse_input(
            [0.1, 0.2, 0.1, 0.2, 0.3], N, K, None, None, *tuple([None] * 5)
        )
    excinfo.match(
        "Dimension mismatch between `c0` and inputted chemical species via `N`."
    )

    with pytest.raises(ValueError) as excinfo:
        eqtk.parsers.parse_input([], None, None, A, G, *tuple([None] * 5))
    excinfo.match("Dimension mismatch between `c0` and the constraint matrix.")

    with pytest.raises(ValueError) as excinfo:
        eqtk.parsers.parse_input(
            [0.1, 0.2, 0.1, 0.2, 0.3], None, None, A, G, *tuple([None] * 5)
        )
    excinfo.match("Dimension mismatch between `c0` and the constraint matrix.")

    # Should raise no exception
    _ = eqtk.parsers.parse_input(
        [0.1, 0.2, 0.1, 0.2, 0.3, 0.1], N, K, None, None, *tuple([None] * 5)
    )
    _ = eqtk.parsers.parse_input(
        [0.1, 0.2, 0.1, 0.2, 0.3, 0.1], None, None, A, G, *tuple([None] * 5)
    )


def test_c0_conversions():
    N = np.array(
        [
            [-1, 0, 1, 0, 0, 0],
            [-1, -1, 0, 1, 0, 0],
            [0, -2, 0, 0, 1, 0],
            [0, -1, -1, 0, 0, 1],
        ]
    )
    K = np.array([0.001, 0.002, 0.003, 0.004])
    target = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    c0 = [1, 2, 3, 4, 5, 6]
    c0, _, _, _, _, _, _, _ = eqtk.parsers.parse_input(
        c0, N, K, None, None, *tuple([None] * 5)
    )
    assert np.array_equal(c0, target)

    c0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    c0, _, _, _, _, _, _, _ = eqtk.parsers.parse_input(
        c0, N, K, None, None, *tuple([None] * 5)
    )
    assert np.array_equal(c0, target)

    c0 = np.array([[1, 2, 3, 4, 5, 6]])
    c0, _, _, _, _, _, _, _ = eqtk.parsers.parse_input(
        c0, N, K, None, None, *tuple([None] * 5)
    )
    assert np.array_equal(c0, target)

    c0 = [[1, 2, 3, 4, 5, 6]]
    c0, _, _, _, _, _, _, _ = eqtk.parsers.parse_input(
        c0, N, K, None, None, *tuple([None] * 5)
    )
    assert np.array_equal(c0, target)


def test_A_rank_deficient():
    A = np.array(
        [
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0, 0.0, 2.0],
            [1.0, 0.0, 0.0, 1.0, 2.0],
        ]
    )
    G = np.zeros(5)
    with pytest.raises(ValueError) as excinfo:
        eqtk.parsers.parse_input(
            [0.1, 0.2, 0.1, 0.2, 0.3], None, None, A, G, *tuple([None] * 5)
        )
    excinfo.match("`A` must have full row rank.")


def test_reshape_empty_A():
    A = np.array([[]]).reshape((1, 0)).astype(float)
    G = np.array([1.0])
    x0 = np.array([1.0])
    x0, N, K, A, G, names, solvent_density, single_point = eqtk.parsers.parse_input(
        x0, None, None, A, G, *tuple([None] * 5)
    )
    assert A.shape[0] == 0
    assert A.shape[1] == 1


def test_A_negative():
    A = np.array([[1, -1]])
    G = np.ones(2)
    with pytest.raises(ValueError) as excinfo:
        eqtk.parsers.parse_input([1, 1], None, None, A, G, *tuple([None] * 5))
    excinfo.match("`A` must have all nonnegative entries.")

    A = np.array([[0, 0, 1, 1], [1, -1, 0, 1]])
    G = np.ones(4)
    with pytest.raises(ValueError) as excinfo:
        eqtk.parsers.parse_input([1, 1, 1, 1], None, None, A, G, *tuple([None] * 5))
    excinfo.match("`A` must have all nonnegative entries.")


def test_cyclic_reactions():
    N = np.array([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
    K = np.array([100.0, 100.0, 100.0])
    c0 = np.array([2.0, 0.05, 1.0])

    with pytest.raises(ValueError) as excinfo:
        eqtk.parse_input(c0, N, K)
    excinfo.match("Rank deficient stoichiometric matrix `N`.")


def test_water_density():
    T = 293.15
    target_molar = 55.4091403681123
    allowed_units = (
        "M",
        "molar",
        "mM",
        "millimolar",
        "uM",
        "µM",
        "micromolar",
        "nM",
        "nanomolar",
        "pM",
        "picomolar",
    )
    multipliers = (1, 1, 1e3, 1e3, 1e6, 1e6, 1e6, 1e9, 1e9, 1e12, 1e12)

    for units, multiplier in zip(allowed_units, multipliers):
        assert np.isclose(
            eqtk.water_density(T, units), target_molar * multiplier
        )

    T = 310.0
    target_molar = 55.141370764358946

    for units, multiplier in zip(allowed_units, multipliers):
        assert np.isclose(
            eqtk.water_density(T, units), target_molar * multiplier
        )

    assert eqtk.water_density(T, None) == 1.0
    assert eqtk.water_density(T, "") == 1.0


def test_parse_solvent_density():
    assert eqtk.parsers._parse_solvent_density(7.0, 293.15, "M") == 7.0
    assert eqtk.parsers._parse_solvent_density(None, 293.15, None) == 1.0
    assert np.isclose(
        eqtk.parsers._parse_solvent_density(None, 293.15, "M"),
        eqtk.water_density(293.15, "M"),
    )

    with pytest.raises(ValueError) as excinfo:
        eqtk.parsers._parse_solvent_density(7.0, 293.15, None)
    excinfo.match("If `solvent_density` is specified, `units` must also be specified.")


def test_nondimensionalize_NK():
    N = np.array(
        [
            [-1, 0, 1, 0, 0, 0],
            [1, 1, 0, -1, 0, 0],
            [0, 2, 0, 0, -1, 0],
            [0, 1, 1, 0, 0, -1],
        ]
    )
    K = np.array([50.0, 0.1, 0.025, 0.01])
    c0 = np.array([1.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    T = 293.15
    solvent_density = None
    units = None
    c0_nondim, K_nondim, solvent_density = eqtk.parsers._nondimensionalize_NK(
        c0, N, K, T, solvent_density, units
    )
    assert np.array_equal(c0_nondim, c0)
    assert np.allclose(K_nondim, K)
    assert solvent_density == 1.0
