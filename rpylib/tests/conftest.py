"""Configuration file
"""

import random
from operator import itemgetter

import pytest

from rpylib.grid.spatial import CTMCUniformGrid
from rpylib.model.utils import *
from rpylib.product.payoff import Forward
from rpylib.product.product import Product
from rpylib.product.underlying import Spot


def pytest_sessionstart(session):
    """
    Called after the Session object has been created and
    before performing collection and entering the run test loop.
    """
    np.random.seed(123456)
    random.seed(123456)


@pytest.fixture(scope="session")
def data():
    val = {'spot1': 100.0,
           'spot2': 80.0,
           'r': 0.05,
           'd1': 0.02,
           'd2': 0.01,
           'maturity': 1.0/12.0
           }
    return val


@pytest.fixture(scope="session")
def forward(data):
    spot, r, d, maturity = itemgetter('spot1', 'r', 'd1', 'maturity')(data)
    fwd = spot*np.exp((r-d)*maturity)
    strike = fwd
    product = Product(payoff_underlying=Spot(), payoff=Forward(strike=strike), maturity=maturity)
    return product


@pytest.fixture(scope="session")
def bs_model(data):
    spot, r, d = itemgetter('spot1', 'r', 'd1')(data)
    sigma = 0.30
    bs_model = create_exponential_of_levy_model(ModelType.BLACKSCHOLES)(spot=spot, r=r, d=d, sigma=sigma)
    return bs_model


@pytest.fixture(scope="session")
def cgmy_model(data):
    spot, r, d = itemgetter('spot1', 'r', 'd1')(data)
    # CGMY model
    c, g, m, y = 0.04945, 10.0, 8.0, 1.1
    cgmy = create_exponential_of_levy_model(ModelType.CGMY)(spot=spot, r=r, d=d, c=c, g=g, m=m, y=y)
    return cgmy


@pytest.fixture(scope="session")
def hem_model(data):
    spot, r, d = itemgetter('spot1', 'r', 'd1')(data)
    # HEM model
    sigma, p, eta1, eta2, intensity = 0.10, 0.6, 25.0, 40.0, 5.0
    hem = create_exponential_of_levy_model(ModelType.HEM)(spot=spot, r=r, d=d, sigma=sigma, p=p, eta1=eta1, eta2=eta2,
                                                          intensity=intensity)
    return hem


def spatial_grid(model):
    return CTMCUniformGrid(h=0.01, model=model)
