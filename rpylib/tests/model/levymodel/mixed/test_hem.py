"""Testing the HEM model"""

import numpy as np
import pytest
from scipy.integrate import quad

from rpylib.model.levymodel.mixed.hem import HEMParameters, ExponentialOfHEMModel


@pytest.fixture()
def hem_model():
    spot, r, d, sigma = 100.0, 0.05, 0.02, 0.10
    p, eta1, eta2, intensity = 0.6, 25.0, 40.0, 5.0
    parameters = HEMParameters(
        sigma=sigma, p=p, eta1=eta1, eta2=eta2, intensity=intensity
    )
    model = ExponentialOfHEMModel(spot=spot, r=r, d=d, parameters=parameters)
    return model


def test_hem_integrate_xx(hem_model):
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
