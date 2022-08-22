"""Testing the CGMY model"""

import numpy as np
import pytest
from scipy.integrate import quad

from rpylib.model.levymodel.purejump.cgmy import (
    CGMYParameters,
    ExponentialOfCGMYModel,
)


@pytest.fixture()
def cgmy_model():
    spot, r, d = 100, 0.02, 0.00
    c, g, m, y = 0.019, 2, 4, 1.2
    parameters = CGMYParameters(c=c, g=g, m=m, y=y)
    model = ExponentialOfCGMYModel(spot=spot, r=r, d=d, parameters=parameters)
    return model


def test_cgmy_integrate(cgmy_model):
    for a, b in [(0.001, 1), (-1, -0.0005), (0.001, np.inf), (-np.inf, -0.0005)]:
        res1 = quad(lambda x: cgmy_model.levy_triplet.nu(x), a, b)[0]
        res2 = cgmy_model.mass(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)


def test_cgmy_integrate_x(cgmy_model):
    for a, b in [(0.001, 1), (-1, -0.0005), (0.001, np.inf), (-np.inf, -0.0005)]:
        res1 = quad(lambda x: x * cgmy_model.levy_triplet.nu(x), a, b)[0]
        res2 = cgmy_model.levy_triplet.nu.integrate_against_x(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)


def test_cgmy_integrate_xx(cgmy_model):
    for a, b in [
        (0.001, 1),
        (-1, -0.0005),
        (0.001, np.inf),
        (-np.inf, -0.0005),
        (-1, 1),
    ]:
        res1 = quad(lambda x: x * x * cgmy_model.levy_triplet.nu(x), a, b)[0]
        res2 = cgmy_model.levy_triplet.nu.integrate_against_xx(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)
