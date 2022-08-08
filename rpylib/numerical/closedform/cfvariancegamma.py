"""
Closed-Formula for the Variance-Gamma model
"""


import mpmath
import numpy as np
import scipy as sp
import scipy.special

import rpylib


class CFVarianceGammaModel:
    def __init__(self, vg_model: "VarianceGammaModel"):
        if not isinstance(
            vg_model,
            rpylib.model.levymodel.purejump.variancegamma.ExponentialOfVarianceGammaModel,
        ):
            raise ValueError("expected a Variance-Gamma model")

        self.model = vg_model

    def call(self, strike: float, maturity: float):
        """Pricing of a call in the Variance-Gamma model

        :param strike: call strike
        :param maturity: call maturity
        """
        raise NotImplementedError(
            "CFVarianceGammaModel::call something wrong in the implementation, "
            "this needs to be double-checked"
        )

        # params = self.model.parameters
        # r, q, spot = params.r, params.d, self.model.spot
        # sigma, nu, theta = params.sigma, params.nu, params.theta
        #
        # gamma = maturity/nu
        # xi = -theta/sigma**2
        # s = sigma/np.sqrt(1 + 0.5*nu*(theta/sigma)**2)
        # alpha = xi*s
        # c1 = 0.5*nu*(alpha + s)**2
        # c2 = 0.5*nu*alpha**2
        # d = (np.log(spot/strike) + (r - q)*maturity + gamma*np.log((1 - c1)/(1 - c2)))/s
        #
        # x1, x2 = d*np.sqrt((1 - c1)/nu), d*np.sqrt((1 - c2)/nu)
        # y1, y2 = (alpha + s)*np.sqrt(nu/(1 - c1)), alpha*s*np.sqrt(nu/(1 - c2))
        #
        # return spot*self._psi_function(x1, y1, gamma) - strike*np.exp(-r*maturity)*self._psi_function(x2, y2, gamma)

    @staticmethod
    def _phi(a, b, c, x, y):
        """This is the second kind and the degenerate hyper-geometric function of two variables
        see https://en.wikipedia.org/wiki/Humbert_series
        """
        res = mpmath.hyper2d({"m+n": [a], "m": [b]}, {"m+n": [c]}, x, y)
        return float(res)

    @staticmethod
    def _psi_function(a, b, gamma):
        phi = CFVarianceGammaModel._phi
        c = np.abs(a) * np.sqrt(2 + b**2)
        u = b / np.sqrt(2 + b**2)
        sign_a = np.sign(a)
        sqrt_2pi = np.sqrt(2 * np.pi)
        g_gamma = sp.special.gamma(gamma)
        num = c ** (gamma + 0.5) * np.exp(sign_a * c) * (1 + u) ** gamma
        aux1 = phi(gamma, 1 - gamma, 1 + gamma, 0.5 * (1 + u), -sign_a * c * (1 + u))
        aux2 = phi(
            1 + gamma, 1 - gamma, 2 + gamma, 0.5 * (1 + u), -sign_a * c * (1 + u)
        )

        term1 = (
            num / (sqrt_2pi * gamma * g_gamma) * sp.special.kv(gamma + 0.5, c) * aux1
        )
        term2 = (
            sign_a
            * num
            * (1 + u)
            / (sqrt_2pi * (1 + gamma) * g_gamma)
            * sp.special.kv(gamma - 0.5, c)
            * aux2
        )
        term3 = (
            sign_a
            * num
            / (sqrt_2pi * gamma * g_gamma)
            * sp.special.kv(gamma - 0.5, c)
            * aux1
        )

        return term1 - term2 + term3
