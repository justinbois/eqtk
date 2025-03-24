import json
import numpy as np

import eqtk


def check_test_case(tc):
    A = np.array(tc["A"]).transpose()
    G = -np.array(tc["g"])
    x0 = np.array(tc["x0"])

    # Remove infinities
    infs = np.isinf(G)
    A = A[:, ~infs]
    G = G[~infs]
    x0 = x0[~infs]

    x = eqtk.solve(c0=x0, A=A, G=G)
    assert eqtk.eqcheck(x, x0, A=A, G=G)

    x = eqtk.solve(c0=x0, A=A, G=G, normal_A=False)
    assert eqtk.eqcheck(x, x0, A=A, G=G)


def test_nupack_design_failures():
    with open("nupack_concentrations-design-failure.json", "r") as f:
        test_cases = json.load(f)

    for tc in test_cases:
        check_test_case(tc)

    with open("nupack_concentrations-design-failure-2.json", "r") as f:
        test_cases = json.load(f)

    for tc in test_cases:
        check_test_case(tc)
