"""
Closed-Form pricing formulas for Lévy models

    .. note:: The Garreau-Kercheval paper is explained in their paper:
                'A Structural Jump Threshold Framework for Credit Risk'
"""

import numpy as np
import scipy.optimize

from ..tools import interval_I


class CFLevyModel:
    def __init__(self, model: "LevyModel"):
        self.model = model

    def _theta(self, level_a):
        theta = self.model.mass(*interval_I(level_a))
        return theta

    def survival_probability(self, level_a: float, t: float):
        """Pricing a survival probability in the Garreau-Kercheval framework

        :param level_a: threshold a
        :param t: time t
        """
        theta = self._theta(level_a)
        return np.exp(-t * theta)

    def cds_spread(self, level_a: float, recovery_rate: float):
        """Calculation of the single-name CDS spread in the Garreau-Kercheval framework

        :param level_a: threshold a
        :param recovery_rate: CDS recovery rate
        """
        theta = self._theta(level_a)
        return (1 - recovery_rate) * theta

    def implied_cds_threshold(self, cds_spread: float, recovery_rate: float, h0: float):
        if h0 <= 0:
            raise ValueError("expected strictly positive h0")

        def fun(threshold):
            return (
                self.cds_spread(level_a=threshold, recovery_rate=recovery_rate)
                - cds_spread
            )

        a, b = -10, -h0
        res = scipy.optimize.brentq(f=fun, a=a, b=b)
        return res

    def implied_cds_spread(
        self, pv: float, level_a: float, recovery_rate: float, maturity: float
    ):
        theta = self._theta(level_a)
        r = self.model.r
        default_leg = (
            (1 - recovery_rate)
            * (1 - np.exp(-(r + theta) * maturity))
            * theta
            / (r + theta)
        )
        fixed_leg = (1 - np.exp(-(r + theta) * maturity)) / (r + theta)

        def fun(spread):
            return default_leg - spread * fixed_leg - pv

        a, b = (
            -5,
            10,
        )  # `a` is negative because very low number of Monte-Carlo paths might yield to a large variance
        # such that the confidence interval contains 0
        res = scipy.optimize.brentq(f=fun, a=a, b=b)
        return res
