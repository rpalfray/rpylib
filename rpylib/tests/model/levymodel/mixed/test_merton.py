"""Testing the Merton model"""

import numpy as np
import pytest
from scipy.integrate import quad

from rpylib.model.levymodel.mixed.merton import (
    MertonParameters,
    ExponentialOfMertonModel,
)


@pytest.fixture()
def hem_model():
    spot, r, d, sigma = 100.0, 0.05, 0.02, 0.10
    mu_j, sigma_j, intensity = 0.01, 0.05, 5.0
    parameters = MertonParameters(
        sigma=sigma, intensity=intensity, mu_j=mu_j, sigma_j=sigma_j
    )
    model = ExponentialOfMertonModel(spot=spot, r=r, d=d, parameters=parameters)
    return model


def test_merton_integrate(hem_model):
    for a, b in [(0.001, 1), (-1, -0.0005), (0.001, np.inf), (-np.inf, -0.0005)]:
        res1 = quad(lambda x: hem_model.levy_triplet.nu(x), a, b)[0]
        res2 = hem_model.mass(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)


def test_merton_integrate_x(hem_model):
    for a, b in [(0.001, 1), (-1, -0.0005), (0.001, np.inf), (-np.inf, -0.0005)]:
        res1 = quad(lambda x: x * hem_model.levy_triplet.nu(x), a, b)[0]
        res2 = hem_model.levy_triplet.nu.integrate_against_x(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)


def test_merton_integrate_xx(hem_model):
    for a, b in [
        (0.001, 1),
        (-1, -0.0005),
        (0.001, np.inf),
        (-np.inf, -0.0005),
        (-1, 1),
    ]:
        res1 = quad(lambda x: x * x * hem_model.levy_triplet.nu(x), a, b)[0]
        res2 = hem_model.levy_triplet.nu.integrate_against_xx(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)
