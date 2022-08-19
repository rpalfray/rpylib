"""Testing the Lévy copula"""

import numpy as np

from rpylib.distribution.levycopula import ClaytonCopula


def test_clayton_inverse_conditional_distribution():
    copula = ClaytonCopula(theta=0.9, eta=0.6)
    eps1, eps2 = 30.0, -25.0
    x1, x2 = np.array([10.0]), np.array([-5.0])
    u11, u12, u21, u22 = (
        copula.conditional_distribution(eps1, x1),
        copula.conditional_distribution(eps1, x2),
        copula.conditional_distribution(eps2, x1),
        copula.conditional_distribution(eps2, x2),
    )
    test11, test12, test21, test22 = (
        copula.inverse_conditional_distribution(eps1, u11),
        copula.inverse_conditional_distribution(eps1, u12),
        copula.inverse_conditional_distribution(eps2, u21),
        copula.inverse_conditional_distribution(eps2, u22),
    )
    diffs = [x1[0] - test11[0], x2[0] - test12[0], x1[0] - test21[0], x2[0] - test22[0]]

    assert all(abs(diff) < 1e-14 for diff in diffs)
