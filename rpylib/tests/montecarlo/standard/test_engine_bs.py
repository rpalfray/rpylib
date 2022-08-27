import pytest
from rpylib.process.levyprocess import LevyProcess

from rpylib.montecarlo.configuration import ConfigurationStandard
from rpylib.montecarlo.standard.engine import Engine
from rpylib.numerical.closedform.cfblackscholes import CFBlackScholes
from rpylib.product.payoff import Vanilla, PayoffType
from rpylib.product.product import Product
from rpylib.product.underlying import Spot


@pytest.fixture()
def mc_bs_engine(bs_model):
    configuration = ConfigurationStandard(mc_paths=50_000, nb_of_processes=1)
    process = LevyProcess(bs_model)
    mc_engine = Engine(configuration=configuration, process=process)
    return mc_engine


@pytest.fixture()
def call(bs_model):
    maturity = 1.3
    strike = bs_model.spot
    product = Product(
        payoff_underlying=Spot(),
        payoff=Vanilla(strike=strike, payoff_type=PayoffType.CALL),
        maturity=maturity,
    )
    return product


def test_monte_carlo_standard_bs_fwd(bs_model, forward, mc_bs_engine):
    # test that the final price is within the 95% quantile - confidence interval
    strike = forward.payoff.strike
    maturity = forward.maturity
    res = mc_bs_engine.price(product=forward)

    mean = res.price()
    mc_stddev = res.mc_stddev()

    # is mean in [price - alpha*mc_stddev, price + alpha*mc_stddev]?
    alpha = 1.96  # 95% quantile confidence interval
    bs_cf_pricer = CFBlackScholes(bs_model)
    price = bs_cf_pricer.forward(strike=strike, maturity=maturity)

    min_val, max_val = price - alpha * mc_stddev, price + alpha * mc_stddev
    assert min_val < mean < max_val


def test_monte_carlo_standard_bs_call(bs_model, call, mc_bs_engine):
    # test that the final price is within the 95% quantile - confidence interval
    strike = call.payoff.strike
    maturity = call.maturity
    res = mc_bs_engine.price(product=call)

    mean = res.price()
    mc_stddev = res.mc_stddev()

    # is mean in [price - alpha*mc_stddev, price + alpha*mc_stddev]?
    alpha = 1.96  # 95% quantile confidence interval
    bs_cf_pricer = CFBlackScholes(bs_model)
    price = bs_cf_pricer.call(strike=strike, maturity=maturity)

    min_val, max_val = price - alpha * mc_stddev, price + alpha * mc_stddev
    test = min_val < mean < max_val
    assert test
