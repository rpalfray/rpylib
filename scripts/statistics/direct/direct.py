"""Plot the distribution of a jump process simulated by a standard Monte-Carlo engine.
    This is only applies to jump model which can be simulated in closed-form.

    .. note:: this script can be easily adapted as long as the model can be simulated directly and the product can
                be priced via the COS method.
"""

from rpylib.model.utils import *
from rpylib.montecarlo.configuration import ConfigurationStandard
from rpylib.montecarlo.standard.engine import Engine
from rpylib.numerical.cosmethod import COSPricer
from rpylib.process.levyprocess import LevyProcess
from rpylib.product.payoff import Vanilla, PayoffType, Forward
from rpylib.product.product import Product, ControlVariates
from rpylib.product.underlying import Spot
from rpylib.tools.timer import timer


@timer
def statistic_standard(model: ExponentialOfLevyModel, mc_paths: int = 10_000, maturity: float = 3/12,
                       percentage_strikes: list[float] = None):
    # Product
    strikes = model.spot*np.array(percentage_strikes or [1.0])
    product = Product(payoff_underlying=Spot(), payoff=Vanilla(strike=strikes, payoff_type=PayoffType.CALL),
                      maturity=maturity)

    # Control Variate
    forward = Product(payoff_underlying=Spot(), payoff=Forward(strike=strikes), maturity=maturity)
    control_variate = ControlVariates([forward], [COSPricer(model).price(product=forward)])

    # Monte-Carlo engine
    configuration = ConfigurationStandard(mc_paths=mc_paths, control_variates=control_variate,
                                          activate_spot_statistics=True, nb_of_processes=1)
    process = LevyProcess(model)
    mc_engine = Engine(configuration=configuration, process=process)

    # Pricing
    result = mc_engine.price(product=product)

    # Results
    mc_price, mc_stddev = result.price(), result.mc_stddev()
    mc_price_no_cv, mc_stddev_no_cv = result.price(no_control_variates=True), result.mc_stddev(no_control_variates=True)
    result.plot_spot_density()
    price = COSPricer(model).price(product=product)

    print('strikes:           ', '* '.join('{:>6.2f} '.format(strike) for strike in strikes))
    print('theoretical price: ', '* '.join('{:>6.3f} '.format(p) for p in price))
    print('mc price         : ', '* '.join('{:>6.3f} '.format(p) for p in np.array(mc_price, ndmin=1)))
    print('mc price / no CV : ', '* '.join('{:>6.3f} '.format(p) for p in np.array(mc_price_no_cv, ndmin=1)))
    print('mc stddev        : ', '* '.join('{:>6.3f} '.format(p) for p in np.array(mc_stddev, ndmin=1)))
    print('mc stddev / no CV: ', '* '.join('{:>6.3f} '.format(p) for p in np.array(mc_stddev_no_cv, ndmin=1)))


if __name__ == '__main__':
    my_model = create_exponential_of_levy_model(ModelType.HEM)()
    my_percentage_strikes = [0.80, 0.90, 1.0, 1.10, 1.20]
    statistic_standard(model=my_model, percentage_strikes=my_percentage_strikes)
