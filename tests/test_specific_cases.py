import numpy as np
import eqtk


def test_binary_binding_3():
    N = np.array([[-1, -1, 1]])
    K = np.array([2e3])
    A = np.array([[1, 0, 1], [0, 1, 1]])
    G = np.array([0, 0, -np.log(K[0])])
    x0 = np.array([3e-3, 5e-3, 0.0])
    x = eqtk.solve(c0=x0, N=N, K=K, units="M")

    assert eqtk.eqcheck(x, x0, N, K, units="M")


def test_binary_binding_5():
    N = np.array([[-2, 0, 1, 0, 0], [-1, -1, 0, 1, 0], [0, -2, 0, 0, 1]])
    K = np.array([2e4, 3e4, 8e3])
    x0_1 = np.array([3e-4, 5e-5, 0, 0, 0])
    x0_2 = np.array([0, 0, 1.5e-4, 0, 2.5e-5])
    x0_3 = np.array([7e-5, 1e-5, 1.0e-4, 3e-5, 5e-6])
    x0 = np.vstack((x0_1, x0_2, x0_3))
    A = np.array([[1, 0, 2, 1, 0], [0, 1, 0, 1, 2]])

    x = eqtk.solve(c0=x0, N=N, K=K, units="M")

    assert eqtk.eqcheck(x, x0, N=N, K=K, units="M")


def test_competition():
    N = np.array([[1, 1, 0, -1, 0], [1, 0, 1, 0, -1]])
    K = np.array([0.05, 0.01])
    x0 = np.array([2.0, 0.05, 1.0, 0.0, 0.0])

    x = eqtk.solve(c0=x0, N=N, K=K, units="µM")

    assert eqtk.eqcheck(x, x0, N, K)


def test_sequential_binding():
    N = np.array([[-1, -1, 1, 0, 0], [0, -1, -1, 1, 0], [0, -1, 0, -1, 1]])
    K = np.array([50.0, 10.0, 40.0])
    c0 = np.array([0.006, 0.003, 0.0, 0.0, 0.0])
    titrated_species = 1
    c0_titrated = np.linspace(0.0, 0.01, 50)

    c0 = np.array([c0 for _ in range(len(c0_titrated))])
    c0[:, titrated_species] = c0_titrated

    c = eqtk.solve(c0, N=N, K=K, units="M")

    assert eqtk.eqcheck(c, c0, N=N, K=K)


def test_aspartic_acid_titration():
    N = np.array(
        [
            [1, 0, -1, 1, 0, 0],
            [1, 0, 0, -1, 1, 0],
            [1, 0, 0, 0, -1, 1],
            [1, 1, 0, 0, 0, 0],
        ]
    )
    K = np.array([10 ** (-1.99), 10 ** (-3.9), 10 ** (-10.002), 1e-14])
    c0 = np.array([0.0, 0.0, 0.001, 0.0, 0.0, 0.0])
    c0_titrant = np.array([0.0, 0.001, 0.0, 0.0, 0.0, 0.0])
    vol_titrant = np.linspace(0.0, 4.0, 100)
    c = eqtk.volumetric_titration(c0, c0_titrant, vol_titrant, N=N, K=K, units="M")

    new_c0 = eqtk.solvers._volumetric_to_c0(c0, c0_titrant, vol_titrant)

    assert eqtk.eqcheck(c, new_c0, N=N, K=K, units="M")


def test_phosphoric_acid_titration():
    N = np.array(
        [
            [1, 0, -1, 1, 0, 0],
            [1, 0, 0, -1, 1, 0],
            [1, 0, 0, 0, -1, 1],
            [1, 1, 0, 0, 0, 0],
        ]
    )
    K = np.array([10 ** (-2.15), 10 ** (-7.2), 10 ** (-12.15), 1e-14])
    c0 = np.array([0.0, 0.0, 0.001, 0.0, 0.0, 0.0])
    c0_titrant = np.array([0.0, 0.001, 0.0, 0.0, 0.0, 0.0])
    vol_titrant = np.linspace(0.0, 4.0, 100)
    c = eqtk.volumetric_titration(c0, c0_titrant, vol_titrant, N=N, K=K, units="M")

    new_c0 = eqtk.solvers._volumetric_to_c0(c0, c0_titrant, vol_titrant)

    assert eqtk.eqcheck(c, new_c0, N=N, K=K, units="M")


def test_exponential_chain():
    n_rxns = 20
    N = np.diag(np.ones(n_rxns), 1)[:-1, :]
    np.fill_diagonal(N, -1)
    N[0, 0] = -2
    N[1:, 0] = -1
    K = 100.0 * np.ones(n_rxns)
    c0 = np.zeros((50, n_rxns + 1))
    c0[:, 0] = np.linspace(0.0, 10.0, 50)

    c = eqtk.solve(c0, N=N, K=K, units="M")

    assert eqtk.eqcheck(c, c0, N=N, K=K, units="M")


def test_example_1():
    N = np.array(
        [
            [-1, 0, 1, 0, 0, 0],
            [-1, -1, 0, 1, 0, 0],
            [0, -2, 0, 0, 1, 0],
            [0, -1, -1, 0, 0, 1],
        ]
    )
    K = np.array([50.0, 10.0, 40.0, 100.0])
    x0 = np.array(
        [[0.001, 0.003, 0.0, 0.0, 0.0, 0.0], [0.001, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )

    x = eqtk.solve(c0=x0, N=N, K=K, units="M")

    assert eqtk.eqcheck(x, x0, N, K, units="M")


def test_example_2():
    N = np.array(
        [
            [-1, -1, -1, 1, 1, 0, 0],
            [-1, 0, 0, 0, 0, -1, 1],
            [-1, 1, 1, 0, -1, 0, 0],
            [0, 1, -1, 0, -1, 1, 0],
        ]
    )
    K = np.array([1.0, 2.0, 3.0, 5.0])
    x0 = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    x = eqtk.solve(c0=x0, N=N, K=K, units="M")

    assert eqtk.eqcheck(x, x0, N, K, units="M")


def test_example_3():
    N = np.array(
        [
            [-1, -1, -1, 1, 1, 0, 0],
            [-1, 0, 0, 0, 0, -1, 1],
            [-1, 1, 1, 0, -1, 0, 0],
            [0, 1, -1, 0, -1, 1, 0],
        ]
    )
    K = np.array([1.0, 2.0, 3.0, 5.0])
    x0 = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    x = eqtk.solve(c0=x0, N=N, K=K, units="M")

    assert eqtk.eqcheck(x, x0, N, K, units="M")


def test_example_4():
    rxns = """
    AB <=> A + B ; 0.015
    AC <=> A + C ; 0.003
    """
    N = eqtk.parse_rxns(rxns)
    c0 = {"A": 1.0, "B": 0.5, "C": 0.25, "AB": 0, "AC": 0}
    c = eqtk.solve(c0, N, units="mM")

    assert eqtk.eqcheck(c, c0, N=N, units="mM")
