"""
Black-Scholes model with constant volatility sigma
"""

import numpy as np

from ..exponentialoflevymodel import ExponentialOfLevyModel
from ...model import Parameters, ModelType
from ....model.levymodel.levymodel import (
    LevyMeasure,
    LevyModel,
    LevyTriplet,
    LevyRepresentation,
    Cumulant,
)
from ....numerical.closedform.cfblackscholes import CFBlackScholes
from ....tools.parameter import positive


class BlackScholesParameters(Parameters):
    """Black-Scholes parameters"""

    sigma = positive("sigma")

    def __init__(self, sigma: float):
        """
        :param sigma: Black-Scholes volatility
        """
        self.sigma = sigma
        self.variance = sigma * sigma

    def __repr__(self) -> str:
        return "BlackScholesParameters(sigma={sigma})".format(sigma=self.sigma)


class _LevyMeasureDiffusion(LevyMeasure):
    """Lévy measure object for diffusive models"""

    def __call__(self, x: float) -> float:
        return 0.0

    def jump_of_finite_activity(self) -> bool:
        return True

    def jump_of_finite_variation(self) -> bool:
        return True

    def finite_first_moment(self):
        return True

    def blumenthal_getoor_index(self) -> float:
        return 0.0

    def integrate(self, a: float, b: float) -> float:
        return 0.0

    def integrate_against_x(self, a: float, b: float) -> float:
        return 0.0

    def integrate_against_xx(self, a: float, b: float) -> float:
        return 0.0


class _BlackScholesCumulant(Cumulant):
    def __init__(self, drift: float, parameters: BlackScholesParameters):
        self.drift = drift
        self.parameters = parameters

    def cumulant1(self, t: float) -> float:
        return self.drift * t

    def cumulant2(self, t: float) -> float:
        return self.parameters.variance * t

    def cumulant3(self, t: float) -> float:
        return 0

    def cumulant4(self, t: float) -> float:
        return 0

    def cumulant5(self, t: float) -> float:
        return 0

    def cumulant6(self, t: float) -> float:
        return 0


class PureDiffusiveModel(LevyModel):
    """Pure diffusive model: L_t where L_t is of the form mu*t + sigma*W_t with W a standard Brownian motion"""

    def __init__(self, mu: float, sigma: float):
        levy_triplet = LevyTriplet(
            a=mu,
            sigma=sigma,
            nu=_LevyMeasureDiffusion(),
            representation=LevyRepresentation.ZERO,
        )
        cumulant = _BlackScholesCumulant(
            drift=mu, parameters=BlackScholesParameters(sigma=sigma)
        )
        super().__init__(
            model_type=ModelType.BLACKSCHOLES,
            levy_triplet=levy_triplet,
            cumulant=cumulant,
        )

    def __repr__(self):
        return "PureDiffusiveModel(mu={mu}, sigma={sigma})".format(
            mu=self.levy_triplet.a, sigma=self.levy_triplet.sigma
        )

    def levy_exponent_pure_jump(self, x: complex) -> complex:
        return 0

    def intensity(self) -> float:
        return 0


class BlackScholesModel(ExponentialOfLevyModel):
    """Black-Scholes model"""

    def __init__(
        self, spot: float, r: float, d: float, parameters: BlackScholesParameters
    ):
        diffusive_model = PureDiffusiveModel(mu=0, sigma=parameters.sigma)
        super().__init__(spot=spot, r=r, d=d, levy_model=diffusive_model)
        self.closed_form = CFBlackScholes(self)
        self.parameters = parameters

    def jump_increment(self, n) -> np.array:
        return np.array([])

    def process_drift(self) -> np.array:
        return self.r - self.d - 0.5 * self.parameters.sigma**2

    def levy_exponent_pure_jump(self, x: complex) -> complex:
        return 0
