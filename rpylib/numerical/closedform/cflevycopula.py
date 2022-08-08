"""
Closed-Form pricing formula for Lévy copula models

    .. note:: The Garreau-Kercheval paper is explained in their paper:
                'A Structural Jump Threshold Framework for Credit Risk'
"""

import numpy as np
import scipy.optimize

from ..tools import interval_I


class CFLevyCopulaModel:
    def __init__(self, levy_copula_model: "LevyCopulaModel"):
        self.levy_copula_model = levy_copula_model

    def _theta(self, levels_a):
        dim = self.levy_copula_model.dimension()
        if dim > 3:
            raise NotImplementedError(
                "theta function not implemented for dimension > 3"
            )
        if dim != len(levels_a):
            raise ValueError(
                "Expected one level per margin, got: "
                + str(len(levels_a))
                + "instead of: "
                + str(dim)
            )
        if any(a >= 0 for a in levels_a):
            raise ValueError("All a-levels must be negative")

        marginal_tail_integral = self.levy_copula_model.margin_tail_integral
        diag = np.array(
            [
                model.mass(*interval_I(ai))
                for model, ai in zip(self.levy_copula_model.models, levels_a)
            ]
        )
        lambda_matrix = np.diag(diag)
        for i in range(dim):
            for j in range(i + 1, dim):
                x = levels_a[i], levels_a[j]
                lambda_matrix[i, j] = -marginal_tail_integral(indices=[i, j], x=x)
        theta = np.sum(lambda_matrix)
        if dim == 3:
            theta -= self.levy_copula_model.tail_integrals(x=levels_a)
        return theta

    def survival_probability(self, levels_a: list[float], t: float):
        """Pricing a survival probability in the Garreau-Kercheval framework

        :param levels_a: thresholds a
        :param t: time t
        """
        theta = self._theta(levels_a)
        probability = np.exp(-t * theta)
        return probability

    def first_to_default_par_spread(self, levels_a: list[float], recovery_rate: float):
        """Pricing the FtD CDS spread in the Garreau-Kercheval framework

        :param levels_a: thresholds a
        :param recovery_rate: CSD recovery rate
        """
        theta = self._theta(levels_a)
        return (1 - recovery_rate) * theta

    def implied_cds_spread(
        self, pv: float, level_a: list[float], recovery_rate: float, maturity: float
    ):
        theta = self._theta(level_a)
        r = self.levy_copula_model.models[0].r
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
            -10,
            10,
        )  # `a` is negative because very low number of Monte-Carlo paths might yield to a large variance
        # such that the confidence interval contains 0
        res = scipy.optimize.brentq(f=fun, a=a, b=b)
        return res
