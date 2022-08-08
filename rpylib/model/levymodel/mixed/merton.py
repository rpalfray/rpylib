"""Merton Model from 'Option Prices When Underlying Stock Returns Are Discontinuous' by Merton
"""

import numpy as np
import scipy.special

from ..exponentialoflevymodel import ExponentialOfLevyModel
from ...model import Parameters, ModelType
from ....model.levymodel.levymodel import (
    LevyMeasure,
    LevyModel,
    LevyTriplet,
    LevyRepresentation,
    Cumulant,
)
from ....tools.parameter import positive, strictly_positive


class MertonParameters(Parameters):
    sigma = positive("sigma")
    mu_j = positive("mu_j")
    sigma_j = strictly_positive("sigma_j")
    intensity = positive("intensity")

    def __init__(self, sigma: float, mu_j: float, sigma_j: float, intensity: float):
        self.sigma = sigma
        self.mu_j = mu_j
        self.sigma_j = sigma_j
        self.intensity = intensity

    def __repr__(self) -> str:
        return "MertonParameters(sigma={sigma}, mu_j={mu_j}, sigma_j={sigma_j}, intensity={intensity})".format(
            sigma=self.sigma,
            mu_j=self.mu_j,
            sigma_j=self.sigma_j,
            intensity=self.intensity,
        )


class _MertonLevyMeasure(LevyMeasure):
    def __init__(self, parameters: MertonParameters):
        self.parameters = parameters

    def __call__(self, x: np.array) -> np.array:
        mu_j, sigma_j = self.parameters.mu_j, self.parameters.sigma_j
        factor = self.parameters.intensity / (sigma_j * np.sqrt(2 * np.pi))
        return factor * np.exp(-((x - mu_j) ** 2) / (2 * sigma_j**2))

    def jump_of_finite_activity(self) -> bool:
        return True

    def jump_of_finite_variation(self) -> bool:
        return True

    def finite_first_moment(self):
        return True

    def blumenthal_getoor_index(self) -> float:
        return 0.0

    @staticmethod
    def _helper_erf_aux(mu, sigma, x):
        return scipy.special.erf((x - mu) / (sigma * np.sqrt(2)))

    def integrate(self, a: float, b: float) -> float:
        mu_j, sigma_j, intensity = (
            self.parameters.mu_j,
            self.parameters.sigma_j,
            self.parameters.intensity,
        )
        return (
            0.5
            * intensity
            * (
                self._helper_erf_aux(mu_j, sigma_j, b)
                - self._helper_erf_aux(mu_j, sigma_j, a)
            )
        )

    def integrate_against_x(self, a: float, b: float) -> float:
        mu_j, sigma_j, intensity = (
            self.parameters.mu_j,
            self.parameters.sigma_j,
            self.parameters.intensity,
        )

        def fun_aux(x):
            return 0.5 * mu_j * self._helper_erf_aux(mu_j, sigma_j, x) - sigma_j / (
                np.sqrt(2 * np.pi)
            ) * np.exp(-((x - mu_j) ** 2) / (2 * sigma_j**2))

        return intensity * (fun_aux(b) - fun_aux(a))

    def integrate_against_xx(self, a: float, b: float) -> float:
        mu_j, sigma_j, intensity = (
            self.parameters.mu_j,
            self.parameters.sigma_j,
            self.parameters.intensity,
        )

        def fun_aux(x):
            first_term = (
                0.5
                * (mu_j**2 + sigma_j**2)
                * self._helper_erf_aux(mu_j, sigma_j, x)
            )
            if x == np.inf or x == -np.inf:
                return first_term
            else:
                return first_term - sigma_j / (np.sqrt(2 * np.pi)) * (
                    mu_j + x
                ) * np.exp(-((x - mu_j) ** 2) / (2 * sigma_j**2))

        return intensity * (fun_aux(b) - fun_aux(a))


class _MertonCumulant(Cumulant):
    def __init__(self, drift: float, parameters: MertonParameters):
        self.drift = drift
        self.parameters = parameters

    def cumulant1(self, t: float) -> float:
        mu_j, intensity = self.parameters.mu_j, self.parameters.intensity
        return (self.drift + intensity * mu_j) * t

    def cumulant2(self, t: float) -> float:
        params = self.parameters
        sigma = params.sigma
        mu_j, sigma_j, intensity = params.mu_j, params.sigma_j, params.intensity
        return (sigma**2 + intensity * (mu_j**2 + sigma_j**2)) * t

    def cumulant4(self, t: float) -> float:
        params = self.parameters
        mu_j, sigma_j, intensity = params.mu_j, params.sigma_j, params.intensity
        return (
            intensity
            * (mu_j**4 + 3 * sigma_j**4 + 6 * mu_j**2 * sigma_j**2)
            * t
        )

    def cumulant6(self, t: float) -> float:
        params = self.parameters
        mu_j, sigma_j, intensity = params.mu_j, params.sigma_j, params.intensity
        return (
            intensity
            * (
                45 * sigma_j**4 * mu_j**2
                + 15 * sigma_j**2 * mu_j**4
                + mu_j**6
                + 15 * sigma_j**6
            )
            * t
        )


class MertonModel(LevyModel):
    def __init__(self, parameters: MertonParameters):
        self.parameters = parameters
        a = -parameters.intensity * parameters.mu_j
        cumulant = _MertonCumulant(parameters=parameters, drift=a)
        levy_triplet = LevyTriplet(
            a=a,
            sigma=parameters.sigma,
            nu=_MertonLevyMeasure(parameters),
            representation=LevyRepresentation.ZERO,
        )
        super().__init__(
            model_type=ModelType.MERTON, levy_triplet=levy_triplet, cumulant=cumulant
        )

    def __repr__(self) -> str:
        return "MertonModel(parameters={parameters})".format(
            parameters=repr(self.parameters)
        )

    def levy_exponent_pure_jump(self, x: complex) -> complex:
        params = self.parameters
        mu_j, sigma_j, intensity = params.mu_j, params.sigma_j, params.intensity
        return intensity * (np.exp(mu_j * x + 0.5 * (sigma_j * x) ** 2) - 1)

    def intensity(self) -> float:
        return self.parameters.intensity

    def jump_increment(self, n) -> np.array:
        z = np.random.normal(
            loc=self.parameters.mu_j, scale=self.parameters.sigma_j, size=n
        )
        return z


class ExponentialOfMertonModel(ExponentialOfLevyModel):
    def __init__(self, spot: float, r: float, d: float, parameters: MertonParameters):
        merton_model = MertonModel(parameters=parameters)
        super().__init__(spot=spot, r=r, d=d, levy_model=merton_model)

        sigma, mu_j, sigma_j, intensity = (
            parameters.sigma,
            parameters.mu_j,
            parameters.sigma_j,
            parameters.intensity,
        )
        self._process_drift = (
            r
            - d
            - 0.5 * sigma**2
            - intensity * (np.exp(mu_j + 0.5 * sigma_j**2) - 1)
        )

    def process_drift(self) -> np.array:
        return self._process_drift
