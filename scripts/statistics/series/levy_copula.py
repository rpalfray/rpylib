"""Analysis of the distribution given by the series representation for a Levy copula model
"""

from rpylib.model.utils import *
from rpylib.montecarlo.configuration import ConfigurationStandard
from rpylib.montecarlo.standard.engine import Engine
from rpylib.numerical.cosmethod import COSPricer
from rpylib.process.levycopulaseries import LevyCopula2dSeriesRepresentation
from rpylib.product.payoff import Vanilla, PayoffType
from rpylib.product.product import Product, ControlVariates
from rpylib.product.underlying import MaximumOfPerformances, NthSpot
from rpylib.tools.timer import timer


@timer
def statistic_series_levy_copula(
    levy_copula_model: LevyCopulaModel,
    tau: float = 1_000,
    maturity: float = 1 / 2,
    mc_paths: int = 10_000,
) -> None:
    """Monte-Carlo pricing of a Best-of option applied to a Levy copula model

    :param levy_copula_model: Levy copula model
    :param tau: cut-off for the series representation
    :param maturity: maturity of the option
    :param mc_paths: number of Monte-Carlo paths
    :return: plot the density of the marginal distributions
    """

    # product
    spot1, spot2 = levy_copula_model.models[0].spot, levy_copula_model.models[1].spot
    product = Product(
        payoff_underlying=MaximumOfPerformances([spot1, spot2]),
        payoff=Vanilla(strike=1.0, payoff_type=PayoffType.CALL),
        maturity=maturity,
    )

    # control variates
    call1 = Product(
        payoff_underlying=NthSpot(1),
        payoff=Vanilla(strike=spot1, payoff_type=PayoffType.CALL),
        maturity=maturity,
    )
    call2 = Product(
        payoff_underlying=NthSpot(2),
        payoff=Vanilla(strike=spot2, payoff_type=PayoffType.CALL),
        maturity=maturity,
    )
    price_calls = [
        COSPricer(levy_copula_model.models[0]).price(product=call1),
        COSPricer(levy_copula_model.models[1]).price(product=call2),
    ]
    control_variates = ControlVariates([call1, call2], price_calls)

    # mc engine
    configuration = ConfigurationStandard(
        mc_paths=mc_paths,
        control_variates=control_variates,
        activate_spot_statistics=True,
    )
    process = LevyCopula2dSeriesRepresentation(
        levy_copula_model=levy_copula_model, tau=tau
    )
    mc_engine = Engine(configuration=configuration, process=process)

    result = mc_engine.price(product=product)
    result.plot_spot_density()

    # results
    mc_price, mc_stddev = result.price(), result.mc_stddev()
    mc_price_no_cv, mc_stddev_no_cv = result.price(
        no_control_variates=True
    ), result.mc_stddev(no_control_variates=True)

    print("{:<20}{:4f}".format("mc price:         ", mc_price))
    print("{:<20}{:4f}".format("mc price/no cv:   ", mc_price_no_cv))
    print("{:<20}{:4f}".format("mc stddev:        ", mc_stddev))
    print("{:<20}{:4f}".format("mc stddev/no cv:  ", mc_stddev_no_cv))


if __name__ == "__main__":
    r, d = 0.03, 0.00
    model1 = create_exponential_of_levy_model(ModelType.VG)(
        spot=1.0, r=r, d=d, sigma=0.30, nu=0.04, theta=-0.2
    )
    model2 = create_exponential_of_levy_model(ModelType.VG)(
        spot=1.0, r=r, d=d, sigma=0.25, nu=0.04, theta=-0.2
    )
    my_maturity = 1 / 12
    my_tau = 1_000
    my_mc_paths = 1_000
    copula = ClaytonCopula(theta=10, eta=0.75)
    my_levy_copula_model = LevyCopulaModel(models=[model1, model2], copula=copula)

    statistic_series_levy_copula(
        levy_copula_model=my_levy_copula_model,
        tau=my_tau,
        maturity=my_maturity,
        mc_paths=my_mc_paths,
    )
