"""HEM Model from the paper 'Pricing Asian Options Under a Hyper-Exponential Jump Diffusion Model' in by Cai and Kou

    .. note:: HEM stands for Hyper-exponential jump model
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
from ....tools.parameter import positive, strictly_positive


class HEMParameters(Parameters):
    sigma = positive("sigma")
    p = strictly_positive("p")
    eta1 = strictly_positive("eta1")
    eta2 = strictly_positive("eta2")
    intensity = positive("intensity")

    def __init__(
        self, sigma: float, p: float, eta1: float, eta2: float, intensity: float
    ):
        self.sigma = sigma
        self.p = p
        self.eta1 = eta1
        self.eta2 = eta2
        self.intensity = intensity

        self._xi = p * eta1 / (eta1 - 1) + (1 - p) * eta2 / (eta2 + 1) - 1

    def __repr__(self) -> str:
        return "HEMParameters(sigma={sigma}, eta1={eta1}, eta2={eta2}, intensity={intensity})".format(
            sigma=self.sigma, eta1=self.eta1, eta2=self.eta2, intensity=self.intensity
        )

    def initialisation(self):
        self._xi = (
            self.p * self.eta1 / (self.eta1 - 1)
            + (1 - self.p) * self.eta2 / (self.eta2 + 1)
            - 1
        )


class _HEMLevyMeasure(LevyMeasure):
    def __init__(self, parameters: HEMParameters):
        self.parameters = parameters

    def __call__(self, x: np.array) -> np.array:
        p, eta1, eta2 = self.parameters.p, self.parameters.eta1, self.parameters.eta2
        a, b = 0, 0
        if x < 0:
            b = (1 - p) * eta2 * np.exp(eta2 * x)
        if x > 0:
            a = p * eta1 * np.exp(-eta1 * x)
        return self.parameters.intensity * (a + b)

    def jump_of_finite_activity(self) -> bool:
        return True

    def jump_of_finite_variation(self) -> bool:
        return True

    def finite_first_moment(self):
        return True

    def blumenthal_getoor_index(self) -> float:
        return 0.0

    def integrate(self, a: float, b: float) -> float:
        params = self.parameters
        intensity, p = params.intensity, params.p

        if a > b:
            raise ValueError(
                "expected a<=b when integrating the levy measure of the HEM model"
            )

        if b <= 0:
            eta2 = params.eta2
            return intensity * (1 - p) * (np.exp(eta2 * b) - np.exp(eta2 * a))
        elif a >= 0:
            eta1 = params.eta1
            return -intensity * p * (np.exp(-eta1 * b) - np.exp(-eta1 * a))
        else:
            return self.integrate(a, 0.0) + self.integrate(0.0, b)

    def integrate_against_x(self, a: float, b: float) -> float:
        params = self.parameters
        intensity, p, eta1, eta2 = params.intensity, params.p, params.eta1, params.eta2

        if a > b:
            raise ValueError(
                "expected a<=b when integrating the levy measure of the HEM model"
            )

        if b <= 0:
            if a == -np.inf:
                return intensity * (1 - p) * np.exp(eta2 * b) * (b - 1 / eta2)
            else:
                return (
                    intensity
                    * (1 - p)
                    * (
                        np.exp(eta2 * b) * (b - 1 / eta2)
                        - np.exp(eta2 * a) * (a - 1 / eta2)
                    )
                )
        elif a >= 0:
            if b == np.inf:
                return intensity * p * np.exp(-eta1 * a) * (a + 1 / eta1)
            else:
                return (
                    -intensity
                    * p
                    * (
                        np.exp(-eta1 * b) * (b + 1 / eta1)
                        - np.exp(-eta1 * a) * (a + 1 / eta1)
                    )
                )
        else:
            return self.integrate_against_x(a, 0.0) + self.integrate_against_x(0.0, b)

    def integrate_against_xx(self, a: float, b: float) -> float:
        params = self.parameters
        intensity, p, eta1, eta2 = params.intensity, params.p, params.eta1, params.eta2

        if a > b:
            raise ValueError(
                "expected a<=b when integrating the levy measure of the HEM model"
            )

        if b <= 0:
            b_eta2 = b * eta2
            b_term = np.exp(b_eta2) * (b_eta2 * (b_eta2 - 2) + 2)
            if a == -np.inf:
                return intensity * (1 - p) * b_term / eta2**2
            else:
                a_eta2 = a * eta2
                return (
                    intensity
                    * (1 - p)
                    * (b_term - np.exp(a_eta2) * (a_eta2 * (a_eta2 - 2) + 2))
                    / eta2**2
                )
        elif a >= 0:
            a_eta1 = a * eta1
            a_term = np.exp(-a_eta1) * (a_eta1 * (a_eta1 + 2) + 2)
            if b == np.inf:
                return intensity * p * a_term / eta1**2
            else:
                b_eta1 = b * eta1
                return (
                    intensity
                    * p
                    * (a_term + np.exp(-b_eta1) * (-b_eta1 * (b_eta1 + 2) - 2))
                    / eta1**2
                )
        else:
            return self.integrate_against_xx(a, 0.0) + self.integrate_against_xx(0.0, b)


class _HEMCumulant(Cumulant):
    def __init__(self, drift: float, parameters: HEMParameters):
        self.drift = drift
        self.parameters = parameters

    def cumulant1(self, t: float) -> float:
        params = self.parameters
        intensity, p, eta1, eta2 = params.intensity, params.p, params.eta1, params.eta2
        return (self.drift + intensity * (p / eta1 - (1 - p) / eta2)) * t

    def cumulant2(self, t: float) -> float:
        params = self.parameters
        sigma = params.sigma
        intensity, p, eta1, eta2 = params.intensity, params.p, params.eta1, params.eta2
        return (sigma**2 + 2 * intensity * (p / eta1**2 + (1 - p) / eta2**2)) * t

    def cumulant4(self, t: float) -> float:
        params = self.parameters
        intensity, p, eta1, eta2 = params.intensity, params.p, params.eta1, params.eta2
        return 24 * intensity * (p / eta1**4 + (1 - p) / eta2**4) * t

    def cumulant6(self, t: float) -> float:
        params = self.parameters
        intensity, p, eta1, eta2 = params.intensity, params.p, params.eta1, params.eta2
        return 720 * intensity * (p / eta1**6 + (1 - p) / eta2**6) * t


class HEMModel(LevyModel):
    def __init__(self, parameters: HEMParameters):
        self.parameters = parameters
        a = -parameters.intensity * (
            parameters.p / parameters.eta1 - (1 - parameters.p) / parameters.eta2
        )
        cumulant = _HEMCumulant(drift=a, parameters=parameters)
        levy_triplet = LevyTriplet(
            a=a,
            sigma=parameters.sigma,
            nu=_HEMLevyMeasure(parameters),
            representation=LevyRepresentation.ZERO,
        )

        super().__init__(
            model_type=ModelType.HEM, levy_triplet=levy_triplet, cumulant=cumulant
        )

    def __repr__(self) -> str:
        return "HEMModel(parameters={parameters})".format(
            parameters=repr(self.parameters)
        )

    def levy_exponent_pure_jump(self, x: complex) -> complex:
        params = self.parameters
        intensity, p, eta1, eta2 = params.intensity, params.p, params.eta1, params.eta2
        return intensity * (p * eta1 / (eta1 - x) + (1 - p) * eta2 / (eta2 + x) - 1)

    def jump_increment(self, n) -> np.array:
        u = np.random.random(size=n)
        v = np.random.random(size=n)
        p, eta1, eta2 = self.parameters.p, self.parameters.eta1, self.parameters.eta2
        wi = np.where(u < p, -1 / eta1, 1 / eta2)
        z = np.log(1 - v) * wi
        return z

    def process_drift(self) -> np.array:
        return -self.parameters.intensity * self.parameters._xi

    def intensity(self) -> float:
        return self.parameters.intensity


class ExponentialOfHEMModel(ExponentialOfLevyModel):
    def __init__(self, spot: float, r: float, d: float, parameters: HEMParameters):
        hem_model = HEMModel(parameters=parameters)
        super().__init__(spot=spot, r=r, d=d, levy_model=hem_model)
        self._process_drift = r - d - parameters.intensity * parameters._xi

    def process_drift(self) -> np.array:
        return self._process_drift
