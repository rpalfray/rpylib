"""Bunch of helper functions to create Lévy models

"""
import copy
from collections import namedtuple
from collections.abc import Callable
from typing import Union

import numpy as np
import scipy.optimize

from .levycopulamodel import LevyCopulaModel
from .levydrivensde.levyforwardmodel import LevyForwardModel
from .levymodel.exponentialoflevymodel import ExponentialOfLevyModel
from .levymodel.levymodel import ModelType, LevyModel
from .levymodel.mixed.blackscholes import (
    BlackScholesParameters,
    PureDiffusiveModel,
    BlackScholesModel,
)
from .levymodel.mixed.hem import HEMParameters, HEMModel, ExponentialOfHEMModel
from .levymodel.mixed.merton import (
    MertonParameters,
    MertonModel,
    ExponentialOfMertonModel,
)
from .levymodel.purejump.cgmy import CGMYParameters, CGMYModel, ExponentialOfCGMYModel
from .levymodel.purejump.variancegamma import (
    VGParameters,
    VarianceGammaModel,
    ExponentialOfVarianceGammaModel,
)
from ..distribution.levycopula import (
    LevyCopula,
    ClaytonCopula,
    IndependentComponentsCopula,
    DependentComponentsCopula,
)
from ..numerical.closedform.cfblackscholes import CFBlackScholes
from ..numerical.cosmethod import COSPricer
from ..product.payoff import Vanilla, PayoffType
from ..product.product import Product
from ..product.underlying import Spot

ModelClasses = namedtuple(
    "ModelClasses", "parameters levy_model exponential_of_levy_model"
)
models_description = {
    ModelType.BLACKSCHOLES: ModelClasses(
        BlackScholesParameters, PureDiffusiveModel, BlackScholesModel
    ),
    ModelType.HEM: ModelClasses(HEMParameters, HEMModel, ExponentialOfHEMModel),
    ModelType.MERTON: ModelClasses(
        MertonParameters, MertonModel, ExponentialOfMertonModel
    ),
    ModelType.CGMY: ModelClasses(CGMYParameters, CGMYModel, ExponentialOfCGMYModel),
    ModelType.VG: ModelClasses(
        VGParameters, VarianceGammaModel, ExponentialOfVarianceGammaModel
    ),
}

LevyModelFunctor = Callable[..., LevyModel]
ExponentialOfLevyModelFunctor = Callable[..., ExponentialOfLevyModel]


def helper_model(
    model_type: ModelType, exponential_of_levy_model: bool = True
) -> Union[LevyModelFunctor, ExponentialOfLevyModelFunctor]:
    """Builder to create a Lévy model

    :param model_type: Lévy model type
    :param exponential_of_levy_model: if true return the helper for the exponential of the Lévy model
    :return: the builder function for the Lévy model or the exponential of the Lévy model if
             exponential_of_levy_model=True
    """
    parameters, model, expofmodel = models_description[model_type]

    def helper(**kwargs):
        return model(parameters=parameters(**kwargs))

    def helper_exp(*, spot: float, r: float, d: float, **kwargs):
        return expofmodel(spot=spot, r=r, d=d, parameters=parameters(**kwargs))

    return helper_exp if exponential_of_levy_model else helper


def _create_hem_model(
    sigma: float = 0.05,
    p: float = 0.6,
    eta1: float = 20,
    eta2: float = 25,
    intensity: float = 3,
):
    return helper_model(ModelType.HEM, False)(
        sigma=sigma, p=p, eta1=eta1, eta2=eta2, intensity=intensity
    )


def _create_merton_model(
    sigma: float = 0.05, sigma_j: float = 0.05, mu_j: float = 0.01, intensity: float = 3
):
    return helper_model(ModelType.MERTON, False)(
        sigma=sigma, sigma_j=sigma_j, mu_j=mu_j, intensity=intensity
    )


def _create_cgmy_model(c: float = 1.0, g: float = 15, m: float = 20, y: float = 0.5):
    return helper_model(ModelType.CGMY, False)(c=c, g=g, m=m, y=y)


def _create_variancegamma_model(
    sigma: float = 0.1, nu: float = 0.06, theta: float = 0.1
):
    return helper_model(ModelType.VG, False)(sigma=sigma, nu=nu, theta=theta)


def _create_blackscholes_model(
    spot: float = 100, r: float = 0.02, d: float = 0.0, sigma: float = 0.10
):
    return helper_model(ModelType.BLACKSCHOLES)(spot=spot, r=r, d=d, sigma=sigma)


def _create_exp_hem_model(
    spot: float = 100,
    r: float = 0.02,
    d: float = 0.0,
    sigma: float = 0.05,
    p: float = 0.6,
    eta1: float = 20,
    eta2: float = 25,
    intensity: float = 3,
):
    return helper_model(ModelType.HEM)(
        spot=spot, r=r, d=d, sigma=sigma, p=p, eta1=eta1, eta2=eta2, intensity=intensity
    )


def _create_exp_merton_model(
    spot: float = 100,
    r: float = 0.02,
    d: float = 0.0,
    sigma: float = 0.05,
    sigma_j: float = 0.05,
    mu_j: float = 0.03,
    intensity: float = 3,
):
    return helper_model(ModelType.MERTON)(
        spot=spot,
        r=r,
        d=d,
        sigma=sigma,
        sigma_j=sigma_j,
        mu_j=mu_j,
        intensity=intensity,
    )


def _create_exp_cgmy_model(
    spot: float = 100,
    r: float = 0.02,
    d: float = 0.0,
    c: float = 1.0,
    g: float = 15,
    m: float = 20,
    y: float = 0.5,
):
    return helper_model(ModelType.CGMY)(spot=spot, r=r, d=d, c=c, g=g, m=m, y=y)


def _create_exp_variancegamma_model(
    spot: float = 100,
    r: float = 0.02,
    d: float = 0.0,
    sigma: float = 0.1,
    nu: float = 0.06,
    theta: float = 0.1,
):
    return helper_model(ModelType.VG)(
        spot=spot, r=r, d=d, sigma=sigma, nu=nu, theta=theta
    )


ModelCreators = namedtuple("ModelCreators", "LevyModel ExponentialOfLevyModel")
models_creators = {
    ModelType.BLACKSCHOLES: ModelCreators(None, _create_blackscholes_model),
    ModelType.HEM: ModelCreators(_create_hem_model, _create_exp_hem_model),
    ModelType.MERTON: ModelCreators(_create_merton_model, _create_exp_merton_model),
    ModelType.CGMY: ModelCreators(_create_cgmy_model, _create_exp_cgmy_model),
    ModelType.VG: ModelCreators(
        _create_variancegamma_model, _create_exp_variancegamma_model
    ),
}


def create_levy_model(model_type: ModelType) -> LevyModelFunctor:
    """Create a Lévy model from the predefined helper function (with default parameter values)
    :param model_type: Lévy model type
    """
    if model_type not in models_creators:
        raise NotImplementedError(
            "No implemented helper function for this Lévy model yet"
        )

    return models_creators[model_type].LevyModel


def create_exponential_of_levy_model(
    model_type: ModelType,
) -> ExponentialOfLevyModelFunctor:
    """Create an exponential of a Lévy model from the predefined helper function (with default parameter values)
    :param model_type: underlying Lévy model type
    """
    if model_type not in models_creators:
        raise NotImplementedError(
            "No implemented helper function for this exponential of Lévy model yet"
        )

    return models_creators[model_type].ExponentialOfLevyModel


def atm_strike(model: LevyModel) -> float:
    """Get the ATM strike (=spot)
    :param model: Lévy model
    :return: the ATM strike
    """
    strike = model.x0_value()
    if isinstance(model, ExponentialOfLevyModel):
        strike = model.spot
    return strike


def calibrate_model_parameter(
    model: ExponentialOfLevyModel,
    parameter: str,
    parameter_interval: tuple,
    product: Product,
    market_price: float,
) -> float:
    """Calibrate one parameter of an exponential of a Lévy model to a given product and its corresponding market price

    :param model: exponential of a Lévy model
    :param parameter: parameter to calibrate
    :param parameter_interval: admissible interval for the calibration of the parameter
    :param product: calibration product
    :param market_price: market price of the calibration product
    :return: the value of the calibrated parameter

        .. note:: one assumes in this function that the product can be priced with the COS method for this model
    """
    calibrated_parameters = copy.deepcopy(model.levy_model.parameters)
    model_cls = type(model)

    def calibration_fun(value: float) -> float:
        calibrated_parameters.__setattr__(parameter, value)
        calibrated_parameters.initialisation()
        calibrated_model = model_cls(
            spot=model.spot, r=model.r, d=model.d, parameters=calibrated_parameters
        )
        # FIXME by design, the constructor of the derived classed of ExponentialOfLévyModel are of the above form
        # but this is `hardcoded` in the sense that this design is not enforced
        price = COSPricer(calibrated_model).price(product=product)
        return price - market_price

    a, b = parameter_interval
    try:
        res = scipy.optimize.brentq(f=calibration_fun, a=a, b=b)
    except ValueError as err:
        raise ValueError(
            "Parameter cannot be calibrated given the market data and its range constraints"
        ) from err

    return res


def calibrate_model_parameter_to_atm_call(
    model: ExponentialOfLevyModel,
    parameter: str,
    parameter_interval: tuple,
    maturity: float,
    bs_sigma: float = 0.10,
) -> float:
    """Calibration of a model parameter to a Call option which market price is given by the Black-Scholes with a given
    Black-Scholes volatility and maturity

    :param model: exponential of a Lévy model
    :param parameter: parameter to calibrate
    :param parameter_interval: admissible interval for the calibration of the parameter
    :param maturity: maturity of the call option
    :param bs_sigma: Black-Scholes volatility of the call option
    :return: the value of the calibrated parameter
    """
    strike = atm_strike(model=model)
    call = Product(
        payoff_underlying=Spot(),
        payoff=Vanilla(strike=strike, payoff_type=PayoffType.CALL),
        maturity=maturity,
    )
    bs_model = create_exponential_of_levy_model(ModelType.BLACKSCHOLES)(
        spot=model.spot, r=model.r, d=model.d, sigma=bs_sigma
    )
    bs_price = CFBlackScholes(bs_model).call(strike=model.spot, maturity=maturity)
    return calibrate_model_parameter(
        model=model,
        parameter=parameter,
        parameter_interval=parameter_interval,
        product=call,
        market_price=bs_price,
    )


DefaultCalibrationConfiguration = namedtuple(
    "DefaultCalibrationConfiguration", "parameter parameter_interval"
)
default_calibration = {
    ModelType.HEM: DefaultCalibrationConfiguration("sigma", (0.0, 1.0)),
    ModelType.MERTON: DefaultCalibrationConfiguration("mu_j", (0.0, 1.0)),
    ModelType.CGMY: DefaultCalibrationConfiguration("c", (1e-12, 20.0)),
    ModelType.VG: DefaultCalibrationConfiguration("sigma", (0.00001, 1.0)),
}


def run_default_calibration(
    model: ExponentialOfLevyModel, maturity: float, bs_sigma: float = 0.10
):
    """The calibration is done on a predefined parameter and applying
    the function :func:`calibrate_model_parameter_to_atm_call`

    :param model: exponential of a Lévy model
    :param maturity: maturity of the call option used in the calibration
    :param bs_sigma: Black-Scholes volatility of the call option
    """
    def_calibration = default_calibration[model.model_type]
    calibrated_parameter = calibrate_model_parameter_to_atm_call(
        model=model,
        parameter=def_calibration.parameter,
        parameter_interval=def_calibration.parameter_interval,
        maturity=maturity,
        bs_sigma=bs_sigma,
    )

    # The 'clean' way to implement this idea (-> build the calibrated model) would be to implement the Observer Pattern
    # (the subject being the Parameter object) but this will do for now as this is only here that we need this feature
    # and the alternative of implementing the Observer Pattern is quite heavy compared to the few lines below
    calibrated_parameters = copy.deepcopy(model.levy_model.parameters)
    calibrated_parameters.__setattr__(def_calibration.parameter, calibrated_parameter)
    calibrated_parameters.initialisation()
    calibrated_model = type(model)(
        spot=model.spot, r=model.r, d=model.d, parameters=calibrated_parameters
    )

    return calibrated_model


def create_clayton_copula(theta: float = 0.7, eta: float = 0.3) -> LevyCopula:
    """Helper to build the Clayton copula function"""
    return ClaytonCopula(theta=theta, eta=eta)


def create_independent_copula() -> LevyCopula:
    """Helper to build the independent copula function"""
    return IndependentComponentsCopula()


def create_dependent_copula() -> LevyCopula:
    """Helper to build the dependent copula function"""
    return DependentComponentsCopula()


def create_levy_copula_model(
    models: [LevyModel], copula: LevyCopula
) -> LevyCopulaModel:
    """Create a Lévy copula model

    :param models: margins
    :param copula: copula function
    """
    return LevyCopulaModel(models=models, copula=copula)


def create_levy_forward_market_model(driver: LevyModel) -> LevyForwardModel:
    """Create a Lévy Forward Market Model with default parameter values"""
    ois_rates = [0.02, 0.02, 0.02, 0.02, 0.02]
    tenors = [5, 6, 7, 8, 9, 10]
    sigma = np.array([[0.50], [0.80], [1.00], [1.25], [1.50]])
    return LevyForwardModel(
        driver=driver, ois_rates=ois_rates, tenors=tenors, sigma=sigma
    )


def create_levy_forward_market_model_copula(driver: [LevyModel]) -> LevyForwardModel:
    """Create a Lévy Forward Market Model with default parameter values
    The Lévy copula is built by taking the drivers input as margins and choosing the Clayton copula with default
    parameters. Other parameters (initial OIS term rates, tenors and sigma volatility matrix) are also hardcoded.

    :param driver: list of Lévy models
    """
    ois_rates = [0.02, 0.02, 0.02, 0.02, 0.02]
    tenors = [5, 6, 7, 8, 9, 10]
    sigma = np.array(
        [[0.50, 1.50], [0.80, 1.25], [1.00, 1.00], [1.25, 0.80], [1.50, 0.50]]
    )
    copula = create_clayton_copula()
    levy_copula = create_levy_copula_model(models=driver, copula=copula)
    return LevyForwardModel(
        driver=levy_copula, ois_rates=ois_rates, tenors=tenors, sigma=sigma
    )
