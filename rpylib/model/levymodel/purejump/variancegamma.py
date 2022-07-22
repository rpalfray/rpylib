"""Variance Gamma model from the paper  `The Variance Gamma Process and Option Pricing` by Carr, Madan and Chang
"""


import numpy as np
import scipy.special as spp

from ..exponentialoflevymodel import ExponentialOfLevyModel
from ...model import Parameters, ModelType
from ....model.levymodel.levymodel import LevyMeasure, LevyModel, LevyTriplet, LevyRepresentation, Cumulant
from ....tools.parameter import positive
from ....tools.integral import integral_xn_exp_minus_x


class VGParameters(Parameters):
    sigma = positive('sigma')

    def __init__(self, sigma: float, nu: float, theta: float):
        self.sigma = sigma
        self.nu = nu
        self.theta = theta

        sigma2 = sigma**2
        self._c = 1/nu
        self._lambda_p = np.sqrt(theta**2 + 2*sigma2/nu)/sigma2 - theta/sigma2
        self._lambda_m = self._lambda_p + 2*theta/sigma2

    def __repr__(self) -> str:
        return 'VGParameters(sigma={sigma}, nu={nu}, theta={theta})'\
            .format(sigma=self.sigma, nu=self.nu, theta=self.theta)

    def initialisation(self):
        sigma, theta, nu = self.sigma, self.theta, self.nu
        sigma2 = sigma**2
        self._c = 1/nu
        self._lambda_p = np.sqrt(theta**2 + 2*sigma2/nu)/sigma2 - theta/sigma2
        self._lambda_m = self._lambda_p + 2*theta/sigma2


class _VGCumulant(Cumulant):

    def __init__(self, drift: float, parameters: VGParameters):
        self.drift = drift
        self.parameters = parameters

    def cumulant1(self, t: float) -> float:
        return (self.drift + self.parameters.theta)*t

    def cumulant2(self, t: float) -> float:
        sigma, nu, theta = self.parameters.sigma, self.parameters.nu, self.parameters.theta
        return (sigma**2 + nu*theta**2)*t

    def cumulant4(self, t: float) -> float:
        sigma, nu, theta = self.parameters.sigma, self.parameters.nu, self.parameters.theta
        sigma2, theta2 = sigma**2, theta**2
        return 3*(sigma2**2*nu + 2*theta2**2*nu**3 + 4*sigma2*theta2*nu**2)*t


class _VGLevyMeasure(LevyMeasure):
    def __init__(self, parameters: VGParameters):
        self.parameters = parameters

    def __call__(self, x: float) -> float:
        c, lm, lp = self.parameters._c, self.parameters._lambda_m, self.parameters._lambda_p
        if x < 0:
            return c*np.exp(-lm*abs(x))/abs(x)
        if x > 0:
            return c*np.exp(-lp*x)/x
        else:
            return 0

    def jump_of_finite_activity(self) -> bool:
        return False

    def jump_of_finite_variation(self) -> bool:
        return True

    def finite_first_moment(self):
        return True

    def blumenthal_getoor_index(self) -> float:
        return 0.0

    def x_nu(self, x: float) -> float:
        c, lm, lp = self.parameters._c, self.parameters._lambda_m, self.parameters._lambda_p
        if x < 0:
            return -c*np.exp(-lm*abs(x))
        if x > 0:
            return c*np.exp(-lp*x)
        else:
            return 0

    def integrate(self, a: float, b: float) -> float:
        c, lm, lp = self.parameters._c, self.parameters._lambda_m, self.parameters._lambda_p
        if b == np.inf:
            if a == np.inf:
                return 0.0
            else:
                return c*spp.exp1(lp*a)

        if a == -np.inf:
            if b == -np.inf:
                return 0.0
            else:
                return c*spp.exp1(-lm*b)

        if a > 0 and b > 0:
            return c*(spp.exp1(lp*a) - spp.exp1(lp*b))
        elif a < 0 and b < 0:
            return c*(spp.exp1(-lm*b) - spp.exp1(-lm*a))
        else:  # a < 0 < b
            return c*(spp.exp1(lp*b) - spp.exp1(-lm*a))

    def integrate_against_x(self, a: float, b: float) -> float:
        c, lm, lp = self.parameters._c, self.parameters._lambda_m, self.parameters._lambda_p

        if a >= 0:
            return c*(np.exp(-lp*a) - np.exp(-lp*b))/lp

        if a < 0 < b:
            return self.integrate_against_x(a, 0.0) + self.integrate_against_x(0.0, b)

        # case b <= 0
        return c*(np.exp(lm*a) - np.exp(lm*b))/lm

    def integrate_against_xx(self, a: float, b: float) -> float:
        c, lm, lp = self.parameters._c, self.parameters._lambda_m, self.parameters._lambda_p

        if a >= 0:
            if b == np.inf:
                return c*(a + 1/lp)*np.exp(-lp*a)
            else:
                return c*((a + 1/lp)*np.exp(-lp*a) - (b + 1/lp)*np.exp(-lp*b))/lp

        if a < 0 < b:
            return self.integrate_against_xx(a, 0.0) + self.integrate_against_xx(0.0, b)

        # case b <= 0
        if a == -np.inf:
            return -c*(b - 1/lm)*np.exp(lm*b)
        else:
            return -c*((b - 1/lm)*np.exp(lm*b) - (a - 1/lm)*np.exp(lm*a))/lm

    def integrate_against_xn(self, a: float, b: float, n: int):
        c = self.parameters._c

        if b < 0:
            lm = self.parameters._lambda_m
            return -c*integral_xn_exp_minus_x(n=n-1, a=a, b=b, alpha=lm)
        else:
            lp = self.parameters._lambda_p
            return c*integral_xn_exp_minus_x(n=n-1, a=a, b=b, alpha=lp)


class VarianceGammaModel(LevyModel):

    def __init__(self, parameters: VGParameters):
        self.parameters = parameters
        triplet = LevyTriplet(a=0.0, sigma=0.0, nu=_VGLevyMeasure(parameters), representation=LevyRepresentation.ZERO)
        super().__init__(model_type=ModelType.VG, levy_triplet=triplet,
                         cumulant=_VGCumulant(drift=0.0, parameters=parameters))

    def __repr__(self) -> str:
        return 'VarianceGammaModel(parameters={parameters})'.format(parameters=repr(self.parameters))

    def levy_exponent_pure_jump(self, x: complex) -> complex:
        sigma, nu, theta = self.parameters.sigma, self.parameters.nu, self.parameters.theta
        res = -np.log(1.0 - 0.5*nu*(x*sigma)**2 - theta*nu*x)/nu
        return res

    def intensity(self) -> float:
        return np.inf


class ExponentialOfVarianceGammaModel(ExponentialOfLevyModel):

    def __init__(self, spot: float, r: float, d: float, parameters: VGParameters):
        vg_model = VarianceGammaModel(parameters=parameters)
        super().__init__(spot=spot, r=r, d=d, levy_model=vg_model)
