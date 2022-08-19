"""Testing the Black-Scholes model"""

import numpy as np
import pytest

from rpylib.model.levymodel.mixed.blackscholes import (
    BlackScholesParameters,
    BlackScholesModel,
)


@pytest.fixture()
def bsmodel():
    spot, r, d, sigma = 90.0, 0.05, 0.01, 0.20
    parameters = BlackScholesParameters(sigma=sigma)
    model = BlackScholesModel(spot=spot, r=r, d=d, parameters=parameters)
    return model


@pytest.mark.parametrize(
    "strike, maturity, expected",
    [
        (100.0, 1.0, -6.018457412646271),
        (100.0, 0.0, -10.0),
        (100.0, np.inf, 0.0),
        (90.0, 1.3, 4.50150044785957),
    ],
)
def test_blackscholes_closedform_forward(bsmodel, strike, maturity, expected):
    result = bsmodel.closed_form.forward(strike=strike, maturity=maturity)

    assert np.isclose(result, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "strike, maturity, forward",
    [
        (100.0, 1.0, -6.018457412646271),
        (100.0, 0.0, -10.0),
        (90.0, 1.3, 4.50150044785957),
    ],
)
def test_blackscholes_closedform_callput_parity(bsmodel, strike, maturity, forward):
    result = (
        bsmodel.closed_form.call(strike=strike, maturity=maturity)
        - bsmodel.closed_form.put(strike=strike, maturity=maturity)
        - forward
    )

    assert np.isclose(result, 0.0, rtol=1e-10, atol=1e-10)
