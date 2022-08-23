"""Analysis of the distribution given by the CTMC scheme for a Levy copula model
"""

from rpylib.grid.spatial import CTMCGridGeometric
from rpylib.model.utils import *
from rpylib.montecarlo.configuration import ConfigurationStandard
from rpylib.montecarlo.standard.engine import Engine
from rpylib.numerical.cosmethod import COSPricer
from rpylib.process.markovchain.markovchainlevycopula import MarkovChainLevyCopula
from rpylib.process.markovchain.markovchain import SamplingMethod
from rpylib.product.payoff import Vanilla, PayoffType
from rpylib.product.product import Product, ControlVariates
from rpylib.product.underlying import MaximumOfPerformances, NthSpot
from rpylib.tools.timer import timer


@timer
def statistic_markov_chain_levy_copula(
    levy_copula_model: LevyCopulaModel, maturity: float = 1 / 2, mc_paths: int = 10_000
) -> None:
    """Monte-Carlo pricing of a Best-of option applied to a Levy copula model

    :param levy_copula_model: Levy copula model
    :param maturity: maturity of the option
    :param mc_paths: number of Monte-Carlo paths
    :return: plot the density of the marginal distributions
    """

    grid = CTMCGridGeometric(
        h=1e-6, nb_of_points_on_each_side=20, model=levy_copula_model
    )

    # product
    spots = [model.spot for model in levy_copula_model.models]
    product = Product(
        payoff_underlying=MaximumOfPerformances(spots),
        payoff=Vanilla(strike=1.0, payoff_type=PayoffType.CALL),
        maturity=maturity,
        notional=100,
    )

    # control variates
    calls = [
        Product(
            payoff_underlying=NthSpot(i),
            payoff=Vanilla(strike=spot, payoff_type=PayoffType.CALL),
            maturity=maturity,
        )
        for i, spot in enumerate(spots, 1)
    ]
    price_calls = [
        COSPricer(model).price(product=call)
        for model, call in zip(levy_copula_model.models, calls)
    ]
    control_variates = ControlVariates(calls, price_calls)

    # mc engine
    configuration = ConfigurationStandard(
        mc_paths=mc_paths,
        control_variates=control_variates,
        activate_spot_statistics=True,
    )
    process = MarkovChainLevyCopula(
        levy_copula_model=levy_copula_model,
        grid=grid,
        method=SamplingMethod.BINARYSEARCHTREEADAPTED,
    )
    mc_engine = Engine(configuration=configuration, process=process)

    result = mc_engine.price(product=product)
    result.plot_spot_density()

    mc_price, mc_stddev = result.price(), result.mc_stddev()
    mc_price_no_cv, mc_stddev_no_cv = result.price(
        no_control_variates=True
    ), result.mc_stddev(no_control_variates=True)

    print("{:<20}{:.10f}".format("mc price:", mc_price))
    print("{:<20}{:.10f}".format("mc price/no cv:", mc_price_no_cv))
    print("{:<20}{: .10f}".format("mc stddev:", mc_stddev))
    print("{:<20}{: .10f}".format("mc stddev/no cv:", mc_stddev_no_cv))


if __name__ == "__main__":
    my_maturity = 1 / 12
    my_mc_paths = 10_000
    model1_ = create_exponential_of_levy_model(ModelType.CGMY)(spot=80, c=0.01, y=0.3)
    model2_ = create_exponential_of_levy_model(ModelType.CGMY)(spot=100, c=10)
    model1 = run_default_calibration(model=model1_, maturity=my_maturity)
    model2 = run_default_calibration(model=model2_, maturity=my_maturity)
    models = [model1, model2]
    copula = ClaytonCopula(theta=0.7, eta=0.3)
    my_levy_copula_model = LevyCopulaModel(models=models, copula=copula)
    statistic_markov_chain_levy_copula(
        levy_copula_model=my_levy_copula_model,
        maturity=my_maturity,
        mc_paths=my_mc_paths,
    )
