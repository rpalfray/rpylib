"""Analysis of the underlying distribution given by the CTMC scheme for a chosen model
"""

import scipy.stats

from rpylib.model.utils import *

from rpylib.distribution.sampling import SamplingMethod
from rpylib.grid.spatial import CTMCGridGeometric
from rpylib.model.levymodel.levymodel import LevyModel
from rpylib.model.levymodel.exponentialoflevymodel import ExponentialOfLevyModel
from rpylib.montecarlo.configuration import ConfigurationStandard
from rpylib.montecarlo.standard.engine import Engine
from rpylib.montecarlo.statistic.statistic import MCStatistics
from rpylib.numerical.cosmethod import COSPricer
from rpylib.product.product import Product
from rpylib.product.payoff import Vanilla, PayoffType
from rpylib.product.underlying import Spot
from rpylib.process.markovchain.markovchain import MarkovChainProcess
from rpylib.tools.timer import timer


def helper_produce_monte_carlo_result(model: LevyModel, maturity, mc_paths) -> tuple[MCStatistics, Product]:
    grid = CTMCGridGeometric(h=1e-7, nb_of_points_on_each_side=30, model=model)

    # product
    strike = atm_strike(model=model)
    product = Product(payoff_underlying=Spot(), payoff=Vanilla(strike=strike, payoff_type=PayoffType.CALL),
                      maturity=maturity)

    # Monte-Carlo engine
    method = SamplingMethod.BINARYSEARCHTREEADAPTED1D
    mc_configuration = ConfigurationStandard(mc_paths=mc_paths, activate_spot_statistics=True)
    process = MarkovChainProcess(model=model, method=method, grid=grid)
    mc_engine = Engine(configuration=mc_configuration, process=process)

    # pricing
    result = mc_engine.price(product=product)
    return result, product


@timer
def statistic_markov_chain(model: ExponentialOfLevyModel, maturity: float = 1/2, mc_paths: int = 10_000):
    """Statistic analysis of the CTMC scheme

    :param model: Lévy model
    :param maturity: maturity of the call option
    :param mc_paths: number of Monte-Carlo paths
    :return: print the pdf implied by the Monte-Carlo engine and the theoretical pdf and also give
             prices given by the Monte-Carlo engine and the COS method (i.e. the theoretical price of the call)
    """
    result, product = helper_produce_monte_carlo_result(model=model, maturity=maturity, mc_paths=mc_paths)

    # results
    mc_price, mc_stddev = result.price(), result.mc_stddev()
    mc_price_no_cv, mc_stddev_no_cv = result.price(no_control_variates=True), result.mc_stddev(no_control_variates=True)
    result.plot_spot_density()
    if isinstance(model, ExponentialOfLevyModel):
        cos_pricer = COSPricer(model=model)
        cos_price = cos_pricer.price(product=product)[0]
        print('{:<20}{:4f}'.format('theoretical price:', cos_price))
    print('{:<20}{:4f}'.format('mc price:         ', mc_price))
    print('{:<20}{:4f}'.format('mc price/no cv:   ', mc_price_no_cv))
    print('{:<20}{:4f}'.format('mc stddev:        ', mc_stddev))
    print('{:<20}{:4f}'.format('mc stddev/no cv:  ', mc_stddev_no_cv))


@timer
def kolmogorov_smirnov_test_markov_chain(model: LevyModel, maturity: float = 1/2, mc_paths: int = 10_000):
    """Kolmogorov-Smirnov test applied to the CTMC

    :param model: Lévy model
    :param maturity: maturity of the call option
    :param mc_paths: number of Monte-Carlo paths
    :return: Kolmogorov-Smirnov statistics
    """
    result, product = helper_produce_monte_carlo_result(model=model, maturity=maturity, mc_paths=mc_paths)

    stats = result._spot_underlying_statistics.stats
    samples = stats.reshape(stats.shape[0])

    def cdf(x):
        return model.cdf(t=maturity, x=x)
    d, p_value = scipy.stats.kstest(samples, cdf)
    print('d={:6f}, p-value={:6f}'.format(d, p_value))

    model.plot_cdf(t=maturity, data=samples, show=True)
    result.plot_spot_density()


if __name__ == '__main__':
    my_maturity = 3/12
    my_mc_paths = 10_000
    my_model = create_exponential_of_levy_model(ModelType.VG)()

    statistic_markov_chain(model=my_model, maturity=my_maturity, mc_paths=my_mc_paths)
