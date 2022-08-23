import pytest
from scipy.integrate import quad
import numpy as np


models_in_scope = pytest.mark.parametrize(
    "model",
    [
        "bs_model",
        "merton_model",
        "hem_model",
        "vg_model",
        "cgmy_model",
    ],
    indirect=True,
)


@models_in_scope
def test_integrate(model, integration_bounds):
    for a, b in integration_bounds:
        res1 = quad(lambda x: model.levy_triplet.nu(x), a, b)[0]
        res2 = model.mass(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)


@models_in_scope
def test_integrate_x(model, integration_bounds):
    for a, b in integration_bounds:
        res1 = quad(lambda x: x * model.levy_triplet.nu(x), a, b)[0]
        res2 = model.levy_triplet.nu.integrate_against_x(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)


@models_in_scope
def test_integrate_xx(model, integration_bounds_xx):
    for a, b in integration_bounds_xx:
        res1 = quad(lambda x: x * x * model.levy_triplet.nu(x), a, b)[0]
        res2 = model.levy_triplet.nu.integrate_against_xx(a, b)
        assert np.isclose(res1, res2, rtol=1e-12, atol=1e-12)
