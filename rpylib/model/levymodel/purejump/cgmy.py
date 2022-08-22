"""CGMY model from the paper `The Fine Structure Of Asset Return` by Carr, Geman, Madan and Yor
"""

import numpy as np
import scipy as sp
import scipy.special
from scipy.integrate import quad

from ..exponentialoflevymodel import ExponentialOfLevyModel
from ...model import Parameters, ModelType
from ....model.levymodel.levymodel import (
    LevyMeasure,
    LevyModel,
    LevyTriplet,
    LevyRepresentation,
    Cumulant,
)
from ....tools.parameter import positive, strictly_positive, strictly_less_than


class CGMYParameters(Parameters):
    c = strictly_positive("c")
    g = positive("g")
    m = positive("m")  # strictly_greater_than(1.0)('M')
    y = strictly_less_than(2.0)("y")

    def __init__(self, c: float, g: float, m: float, y: float):
        self.c = c
        self.g = g
        self.m = m
        self.y = y

        self._CGammamY = c * sp.special.gamma(-y)
        self._MpowerY = np.power(m, y)
        self._GpowerY = np.power(g, y)

    def __repr__(self) -> str:
        return "CGMYParameters(c={c}, g={g}, m={m}, y={y})".format(
            c=self.c, g=self.g, m=self.m, y=self.y
        )

    def initialisation(self):
        c, g, m, y = self.c, self.g, self.m, self.y
        self._CGammamY = c * sp.special.gamma(-y)
        self._MpowerY = np.power(m, y)
        self._GpowerY = np.power(g, y)


class _CGMYCumulant(Cumulant):
    def __init__(self, drift: float, parameters: CGMYParameters):
        self.drift = drift
        self.parameters = parameters

    def cumulant1(self, t: float) -> float:
        drift = self.drift
        c, g, m, y = (
            self.parameters.c,
            self.parameters.g,
            self.parameters.m,
            self.parameters.y,
        )
        if y == 1:
            return drift * t  # FIXME infinite cumulant for Y=1
        return drift * t

    def cumulant2(self, t: float) -> float:
        c, g, m, y = (
            self.parameters.c,
            self.parameters.g,
            self.parameters.m,
            self.parameters.y,
        )
        return c * t * sp.special.gamma(2 - y) * (m ** (y - 2) + g ** (y - 2))

    def cumulant4(self, t: float) -> float:
        c, g, m, y = (
            self.parameters.c,
            self.parameters.g,
            self.parameters.m,
            self.parameters.y,
        )
        return c * t * sp.special.gamma(4 - y) * (m ** (y - 4) + g ** (y - 4))

    def cumulant6(self, t: float) -> float:
        c, g, m, y = (
            self.parameters.c,
            self.parameters.g,
            self.parameters.m,
            self.parameters.y,
        )
        return c * t * sp.special.gamma(6 - y) * (m ** (y - 6) + g ** (y - 6))


class _CGMYLevyMeasure(LevyMeasure):
    def __init__(self, parameters: CGMYParameters):
        self.parameters = parameters

    def __str__(self):
        return "CGMY Levy measure: " + str(self.parameters)

    def __call__(self, x: float) -> float:
        c, y = self.parameters.c, self.parameters.y
        x_bar = abs(x)
        den = np.power(x_bar, y + 1)
        if x < 0:
            return c * np.exp(-self.parameters.g * x_bar) / den
        if x > 0:
            return c * np.exp(-self.parameters.m * x_bar) / den
        return 0

    def jump_of_finite_activity(self) -> bool:
        return self.parameters.y < -1

    def jump_of_finite_variation(self) -> bool:
        return self.parameters.y < 1.0

    def finite_first_moment(self):
        return True

    def blumenthal_getoor_index(self) -> float:
        return max(0.0, self.parameters.y)

    def x_nu(self, x: float) -> float:
        c, y = self.parameters.c, self.parameters.y
        x_bar = abs(x)
        den = np.power(x_bar, y)
        if x < 0:
            return c * np.exp(-self.parameters.g * x_bar) / den * np.sign(x)
        if x > 0:
            return c * np.exp(-self.parameters.m * x_bar) / den
        return 0

    def integrate(self, a: float, b: float) -> float:
        if b == np.inf:
            if a == np.inf:
                return 0.0
            else:
                return self.__integrate_levy_measure_a_to_inf(a)

        if a == -np.inf:
            if b == -np.inf:
                return 0.0
            else:
                return self.__integrate_levy_measure_inf_to_b(b)

        return self.__integrate_levy_measure_a_to_b(a, b)

    def integrate_against_x(self, a: float, b: float) -> float:
        c, g, m, y = (
            self.parameters.c,
            self.parameters.g,
            self.parameters.m,
            self.parameters.y,
        )

        if a >= 0:
            if b == np.inf:
                return c * self.__integrate_h_to_inf_for_xx(alpha=y, h=a, u=m)
            else:
                return c * (
                    self.__integrate_h_to_inf_for_xx(alpha=y, h=a, u=m)
                    - self.__integrate_h_to_inf_for_xx(alpha=y, h=b, u=m)
                )

        if b <= 0:
            if a == -np.inf:
                return -c * self.__integrate_h_to_inf_for_xx(alpha=y, h=-b, u=g)
            else:
                return c * (
                    self.__integrate_h_to_inf_for_xx(alpha=y, h=-a, u=g)
                    - self.__integrate_h_to_inf_for_xx(alpha=y, h=-b, u=g)
                )

        return self.integrate_against_x(a, 0.0) + self.integrate_against_x(0.0, b)

    def integrate_against_xx(self, a: float, b: float) -> float:
        if a < 0 < b:
            c, g, m, y = (
                self.parameters.c,
                self.parameters.g,
                self.parameters.m,
                self.parameters.y,
            )
            cst = c * scipy.special.gamma(2 - y)

            if m == 0:
                first = c * b ** (2 - y) / (2 - y)
            else:
                first = cst * scipy.special.gammainc(2 - y, m * b) / m ** (2 - y)

            if g == 0:
                second = c * abs(a) ** (2 - y) / (2 - y)
            else:
                second = cst * scipy.special.gammainc(2 - y, -g * a) / g ** (2 - y)

            return first + second

        points = None
        if self.parameters.y > 1:
            if a < 0 < b:
                points = [0]

        return quad(self._xx_levy_measure, a, b, points=points, limit=100)[0]

    def _xx_levy_measure(self, x):
        c, g, m, y = (
            self.parameters.c,
            self.parameters.g,
            self.parameters.m,
            self.parameters.y,
        )
        x_bar = abs(x)
        factor = np.power(x_bar, 1 - y)
        if x < 0:
            return c * np.exp(-g * x_bar) * factor
        elif x > 0:
            return c * np.exp(-m * x_bar) * factor
        else:
            return 0

    def __integrate_h_to_inf(self, alpha, h, u):
        """integral(exp(-ux)/pow(x, 1+alpha), x=h...inf)"""
        uh = u * h
        if alpha == 0:
            return scipy.special.exp1(uh)

        expmuh = np.exp(-uh)
        if alpha >= 1:
            return expmuh / (alpha * h**alpha) - (
                u / alpha
            ) * self.__integrate_h_to_inf(alpha=alpha - 1, h=h, u=u)

        g2malpha = scipy.special.gamma(2 - alpha)
        ginccuh = scipy.special.gammaincc(2 - alpha, uh)

        res = (
            expmuh * (1 + uh / (1 - alpha))
            - (uh**alpha) * g2malpha * ginccuh / (1 - alpha)
        ) / (alpha * (h**alpha))

        return res

    def __integrate_levy_measure_a_to_inf(self, a):
        c, m, y = self.parameters.c, self.parameters.m, self.parameters.y
        return c * self.__integrate_h_to_inf(alpha=y, h=a, u=m)

    def __integrate_levy_measure_inf_to_b(self, b):
        c, g, y = self.parameters.c, self.parameters.g, self.parameters.y
        return c * self.__integrate_h_to_inf(alpha=y, h=-b, u=g)

    def __integrate_levy_measure_a_to_b(self, a, b):
        if a >= 0 and b > 0:
            return self.__integrate_levy_measure_a_to_inf(
                a
            ) - self.__integrate_levy_measure_a_to_inf(b)
        elif a < 0 and b <= 0:
            return self.__integrate_levy_measure_inf_to_b(
                b
            ) - self.__integrate_levy_measure_inf_to_b(a)
        else:  # a < 0 < b
            if self.parameters.y > 0:
                return np.inf
            else:
                return self.__integrate_levy_measure_a_to_b(
                    a, 0.0
                ) + self.__integrate_levy_measure_a_to_b(0.0, b)

    @staticmethod
    def __integrate_h_to_inf_for_xx(alpha, h, u):
        """integral(exp(-ux)/pow(x, alpha), x=h...inf)"""
        uh = u * h
        if alpha == 1.0:
            res = scipy.special.exp1(uh)
        else:
            expmuh = np.exp(-uh)
            g2malpha = scipy.special.gamma(2 - alpha)
            ginccuh = scipy.special.gammaincc(2 - alpha, uh)
            res = (
                h ** (1 - alpha) * expmuh - u ** (alpha - 1) * g2malpha * ginccuh
            ) / (alpha - 1)

        return res


class CGMYModel(LevyModel):
    def __init__(self, parameters: CGMYParameters):
        self.parameters = parameters
        cumulant = _CGMYCumulant(drift=0, parameters=parameters)
        if parameters.y < 0.0:
            representation = LevyRepresentation.ZERO
        else:
            representation = LevyRepresentation.CENTER

        triplet = LevyTriplet(
            a=0,
            sigma=0.0,
            nu=_CGMYLevyMeasure(parameters),
            representation=representation,
        )
        super().__init__(
            model_type=ModelType.CGMY, levy_triplet=triplet, cumulant=cumulant
        )

    def __repr__(self):
        return "CGMYModel(parameters={parameters})".format(
            parameters=repr(self.parameters)
        )

    def levy_exponent_pure_jump(self, x: complex) -> complex:
        p = self.parameters
        c, g, m, y = p.c, p.g, p.m, p.y

        res = 0
        if y == 0:
            res += -c * (np.log(1 + x / g) + np.log(1 - x / m))
        elif y == 1.0:
            res += c * (
                (g + x) * np.log(g + x)
                - g * np.log(g)
                + (m - x) * np.log(m - x)
                - m * np.log(m)
            )
        else:
            # adjustment for y >= 0 because of the center representation
            # see for example equation (2.4) in "Monte Carlo option pricing for tempered stable (CGMY) processes"
            # by Poirot and Tankov
            aux_g = x * y * g ** (y - 1)
            aux_m = x * y * m ** (y - 1)
            res += p._CGammamY * (
                np.power(g + x, y)
                - aux_g
                + np.power(m - x, y)
                + aux_m
                - p._GpowerY
                - p._MpowerY
            )

        return res

    def intensity(self) -> float:
        return np.inf


class ExponentialOfCGMYModel(ExponentialOfLevyModel):
    def __init__(self, spot: float, r: float, d: float, parameters: CGMYParameters):
        cgmy_model = CGMYModel(parameters=parameters)
        super().__init__(spot=spot, r=r, d=d, levy_model=cgmy_model)
