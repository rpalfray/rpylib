"""Exponential of a Lévy process, that is the underlying is S modelled as:
    S_t = S_0 exp(L_t) and L_t is a Lévy model
"""

import math
from collections.abc import Callable
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from .levymodel import LevyModel
from ...numerical.cosmethod import COSPricer
from ...process.process import ProcessRepresentation
from ...tools.parameter import positive


class MomentsDecorator:
    """Class decorator that specifies the first standardised moments of the distribution"""

    def __call__(self, cls):
        class ClsWithMoments(cls):
            def std_moment(
                self, moment: [Union[float, list[float]]], t: float
            ) -> float:
                """Standard moment of an exponential of a Lévy model S = S0*exp((r-d)t + L)
                   where L is the Lévy model

                :param moment: moment order
                :param t: time
                :return: the expectation of E[ (S_t/S_0)^k ] where k=moment and Fwd = S_0 exp(mu*t), mu=r-d
                """
                return self.log_characteristic_function(
                    t=t, x=-1j * moment, log_spot=0
                ).real

            def mean(self, t: float) -> float:
                """First moment

                :param t: time
                :return: mean of S_t/S_0
                """
                return self.std_moment(1, t)

            def stddev(self, t: float) -> float:
                """Second standardised moment

                :param t: time
                :return: the standard deviation of S_t/S_0
                """
                m1, m2 = self.std_moment([1, 2], t)
                if m2 < m1**2:
                    raise ValueError("Moments::stddev, negative variance")

                return math.sqrt(m2 - m1**2)

            def skewness(self, t: float) -> float:
                """Third standardized moment

                :param t: time
                :return: the skewness of S_t/S_0
                """
                m1, m3 = self.std_moment([1, 3], t)
                sig = self.stddev(t)
                if sig < 1e-12:
                    return 0
                else:
                    return (m3 - 3 * m1 * sig**2 - m1**3) / sig**3

            def kurtosis(self, t: float) -> float:
                """Fourth standardised moment

                :param t: time
                :return: the kurtosis of S_t/S_0
                """
                m1, m2, m3, m4 = self.std_moment([1, 2, 3, 4], t)
                m1p2 = m1**2
                sig = self.stddev(t)
                if sig < 1e-12:
                    return 0
                else:
                    return (m4 - 4 * m1 * m3 + 6 * m1p2 * m2 - 3 * m1p2**2) / sig**4

        return ClsWithMoments


@MomentsDecorator()
class ExponentialOfLevyModel(LevyModel):
    """Exponential of Time-homogeneous Lévy models

    The model is defined as S_t = S_0 exp((r-q + omega)t + L_t) where L_t is the Lévy process with initial value x0=0.
    r and q are respectively the interest rate and the dividend yield, and omega a 'convexity' adjustment
    such that S_t is martingale.
    """

    # The Lévy model S = S_0 exp(drift*t + L) is simulated via the log-process log(S) = log(S_0) + drift*t + L
    process_representation = ProcessRepresentation.LOG

    spot = positive("spot")
    r = positive("r")  # kept positive for simplicity
    d = positive("d")

    def __init__(self, spot: float, r: float, d: float, levy_model: LevyModel):
        """
        :param spot: underlying spot
        :param r: interest rate
        :param d: dividend rate
        :param levy_model: underlying Lévy model L
        """
        super().__init__(
            model_type=levy_model.model_type,
            levy_triplet=levy_model.levy_triplet,
            cumulant=levy_model.cumulant,
        )
        self.spot = spot
        self.log_spot = np.log(spot)
        self.r = r
        self.d = d
        self.levy_model = levy_model
        self.omega = -levy_model.levy_exponent(x=-1j).real
        # the imaginary part is 0, we use `.real` to enforce the float type

    def __str__(self):
        cls = self.__class__.__name__
        return "{cls}(spot={spot}, r={r}, d={d}, levy_model={lm})".format(
            cls=cls, spot=self.spot, r=self.r, d=self.d, lm=str(self.levy_model)
        )

    def __repr__(self):
        return "ExponentialOfLevyModel(spot={spot}, r={r}, d={d}, levy_model={levy_model})".format(
            spot=self.spot, r=self.r, d=self.d, levy_model=repr(self.levy_model)
        )

    def dimension(self) -> int:
        return self.levy_model.dimension()

    def x0_value(self):
        return self.log_spot

    def intensity(self) -> float:
        return self.levy_model.intensity()

    def drift(self, t: float = 0, x: np.array = 0) -> np.array:
        """:return: the drift coefficient of the exponential of the Lévy process"""
        return self.r - self.d + self.omega

    def jump_increment(self, n) -> np.array:
        return self.levy_model.jump_increment(n=n)

    def df(self, t: float) -> float:
        return np.exp(-self.r * t)

    def log_characteristic_function(
        self, t: float, x: complex, log_spot: float = None
    ) -> complex:
        """:return: the characteristic function of the log process log(S_t), that is
        E[exp(i x log(S_t)] = E[exp(i x (log(S_0) + (r-d+omega)t + L_t)] where L is a Lévy process.
        """
        log_spot_val = log_spot if log_spot is not None else self.log_spot
        drift = self.r - self.d + self.omega
        levy_cf = self.levy_model.characteristic_function(t, x)
        return np.exp(1j * x * (log_spot_val + t * drift)) * levy_cf

    def cdf(self, t: float, x: np.array):
        """Cumulative distribution function of the exponential of the Lévy process"""
        return COSPricer(self, n=2_000, l=20).cdf(time=t, x=x)

    def density(
        self, t
    ) -> Callable[[float], Callable[[Union[float, np.ndarray]], float]]:
        """Density function of the exponential of the Lévy model as implied by the COS method"""

        def helper(s):
            return COSPricer(self).density(time=t, s=s)

        return helper

    def plot_density(self, t: float, show: bool = False) -> None:
        fwd = self.spot * np.exp(self.rd)
        bins = np.arange(start=fwd * 0.1, stop=fwd * 2.0, step=0.01)
        y = self.density(t=t)(bins)
        plt.plot(bins, y, "r--", alpha=0.60)
        if show:
            plt.show()

    def plot_cdf(
        self,
        t: float,
        data: np.array,
        log_normalisation: bool = True,
        show: bool = False,
        title="",
    ) -> None:
        fwd = self.spot * np.exp(self.r - self.d)
        bins = np.arange(start=fwd * 0.2, stop=fwd * 3.5, step=0.01)
        y = self.cdf(t=t, x=bins)
        return LevyModel._helper_plot_cdf(
            fwd, bins, y, log_normalisation, data, title, show
        )
