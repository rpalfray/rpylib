"""Configuration file
"""

from operator import itemgetter

import pytest

from rpylib.grid.spatial import CTMCUniformGrid
from rpylib.model.utils import *
from rpylib.product.payoff import Forward
from rpylib.product.product import Product
from rpylib.product.underlying import Spot


@pytest.fixture(scope="session")
def data():
    val = {
        "spot1": 100.0,
        "spot2": 80.0,
        "r": 0.05,
        "d1": 0.02,
        "d2": 0.01,
        "maturity": 1.0 / 12.0,
    }
    return val


@pytest.fixture(scope="session")
def forward(data):
    spot, r, d, maturity = itemgetter("spot1", "r", "d1", "maturity")(data)
    fwd = spot * np.exp((r - d) * maturity)
    strike = fwd
    product = Product(
        payoff_underlying=Spot(), payoff=Forward(strike=strike), maturity=maturity
    )
    return product


@pytest.fixture
def model(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="session")
def bs_model(data):
    spot, r, d = itemgetter("spot1", "r", "d1")(data)
    sigma = 0.30
    bs_model = create_exponential_of_levy_model(ModelType.BLACKSCHOLES)(
        spot=spot, r=r, d=d, sigma=sigma
    )
    return bs_model


@pytest.fixture(scope="session")
def merton_model(data):
    spot, r, d = itemgetter("spot1", "r", "d1")(data)
    mu_j, sigma_j, sigma, intensity = 0.01, 0.05, 0.10, 5.0
    parameters = MertonParameters(
        sigma=sigma, intensity=intensity, mu_j=mu_j, sigma_j=sigma_j
    )
    model = ExponentialOfMertonModel(spot=spot, r=r, d=d, parameters=parameters)
    return model


@pytest.fixture(scope="session")
def hem_model(data):
    spot, r, d = itemgetter("spot1", "r", "d1")(data)
    sigma, p, eta1, eta2, intensity = 0.10, 0.6, 25.0, 40.0, 5.0
    hem = create_exponential_of_levy_model(ModelType.HEM)(
        spot=spot, r=r, d=d, sigma=sigma, p=p, eta1=eta1, eta2=eta2, intensity=intensity
    )
    return hem


@pytest.fixture(scope="session")
def vg_model(data):
    spot, r, d = itemgetter("spot1", "r", "d1")(data)
    sigma, nu, theta = 0.1, 0.02, 0.1
    parameters = VGParameters(sigma=sigma, nu=nu, theta=theta)
    model = ExponentialOfVarianceGammaModel(spot=spot, r=r, d=d, parameters=parameters)
    return model


@pytest.fixture(scope="session")
def cgmy_model(data):
    spot, r, d = itemgetter("spot1", "r", "d1")(data)
    c, g, m, y = 0.04945, 10.0, 8.0, 1.1
    cgmy = create_exponential_of_levy_model(ModelType.CGMY)(
        spot=spot, r=r, d=d, c=c, g=g, m=m, y=y
    )
    return cgmy


def spatial_grid(model):
    return CTMCUniformGrid(h=0.01, model=model)


@pytest.fixture(scope="session")
def integration_bounds():
    return [(0.001, 1), (-1, -0.0005), (0.001, np.inf), (-np.inf, -0.0005)]


@pytest.fixture(scope="session")
def integration_bounds_xx(integration_bounds):
    return integration_bounds + [(-1, 1)]
