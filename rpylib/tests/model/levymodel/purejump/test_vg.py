"""Testing the Varaiance Gamma model"""

import numpy as np
import pytest
from scipy.integrate import quad

from rpylib.model.levymodel.purejump.variancegamma import (
    VGParameters,
    ExponentialOfVarianceGammaModel,
)


@pytest.fixture()
def vg_model():
    spot, r, d = 100, 0.02, 0.00
    sigma, nu, theta = 0.1, 0.02, 0.1
    parameters = VGParameters(sigma=sigma, nu=nu, theta=theta)
    model = ExponentialOfVarianceGammaModel(spot=spot, r=r, d=d, parameters=parameters)
    return model


def test_vg_integrate(vg_model):
    for a, b in [(0.001, 1), (-1, -0.0005), (0.001, np.inf), (-np.inf, -0.0005)]:
        res1 = quad(lambda x: vg_model.levy_triplet.nu(x), a, b)[0]
        res2 = vg_model.mass(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)


def test_vg_integrate_x(vg_model):
    for a, b in [(0.001, 1), (-1, -0.0005), (0.001, np.inf), (-np.inf, -0.0005)]:
        res1 = quad(lambda x: x * vg_model.levy_triplet.nu(x), a, b)[0]
        res2 = vg_model.levy_triplet.nu.integrate_against_x(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)


def test_vg_integrate_xx(vg_model):
    for a, b in [
        (0.001, 1),
        (-1, -0.0005),
        (0.001, np.inf),
        (-np.inf, -0.0005),
        (-1, 1),
    ]:
        res1 = quad(lambda x: x * x * vg_model.levy_triplet.nu(x), a, b)[0]
        res2 = vg_model.levy_triplet.nu.integrate_against_xx(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)
