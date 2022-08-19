"""Testing the integral module"""

import numpy as np
from scipy.integrate import quad

import pytest

from rpylib.tools.integral import integral_xn_exp_minus_x


@pytest.mark.parametrize("alpha", [0.1, 0.7, 2.4, 7.0])
def test_integral_xn_exp_minus_x(alpha):
    ab = [
        (-np.inf, -8.7),
        (-np.inf, 2.5),
        (-10.3, -8.7),
        (-3.1, -0.6),
        (-0.7, 0.0),
        (-0.7, 2.3),
        (0.0, 2.7),
        (0.4, 2.7),
        (1.3, 50.6),
        (1.3, np.inf),
        (-1.3, np.inf),
    ]

    epsabs = epsrel = 1e-12
    abserr = 1e-11  # we accept the test if the absolute error is less than abserr

    for n in [1]:

        def func(x):
            return x**n * np.exp(-alpha * abs(x))

        for a, b in ab:
            numerical_integral, error = quad(
                func=func, a=a, b=b, epsabs=epsabs, epsrel=epsrel
            )
            exact_integral = integral_xn_exp_minus_x(n=n, a=a, b=b, alpha=alpha)
            assert abs(numerical_integral - exact_integral) < abserr
