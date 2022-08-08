"""
Description of a generic Lévy model: this involves the definition of the Lévy measure (and truncated Lévy measure)
as well as Lévy triplet. Cumulants are also defined for those models which have some.
"""

import abc
from collections.abc import Callable
from enum import Enum
from numbers import Real

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from statsmodels.distributions.empirical_distribution import ECDF

from ..model import Model, Parameters, ModelType
from ...process.process import ProcessRepresentation
from ...tools.parameter import positive


class LevyRepresentation(Enum):
    # :math:`\\Psi(z) = ... + int (exp(izx) - 1 - iz h(x)) nu(dx)`
    # :math:`\\Psi`: characteristic function
    # the default (canonical representation) h function is h(x) = x if |x|<1
    # Given some condition on nu, one can move some "value" of the integral into the drift
    ZERO = 1  # h(x) = 0          -> finite variation case
    CENTER = 2  # h(x) = x          -> finite first moment
    ONEONE = 3  # h(x) = x if |x|<1 -> `canonical` representation
    TILDE = 4  # h(x) = 1 if |x|<V where V=1 if integral(|x|nu(dx), |x|<=1) is finite, else 0


class LevyMeasure:
    """Definition of a Lévy measure and its integration against power of x. The main characteristics of the Lévy measure
    are given directly by the value of its Blumenthal-Getoor index.

    .. todo:: enforce init for subclasses to be __init__(self, parameters: Parameters)
    """

    @abc.abstractmethod
    def __call__(self, x: np.array) -> np.array:
        """Compute the standard levy measure at x"""

    def __str__(self):
        cls = self.__class__.__name__
        return "{}".format(cls)

    def support(self):
        """
        :return: the support of the Lévy measure

        .. note:: in theory, this is really :math:`\\mathbb{R}^*`
        """
        return -np.inf, np.inf

    @abc.abstractmethod
    def jump_of_finite_activity(self) -> bool:
        """:return: true if the integral of :math:`\\nu(dx)` over :math:`\\mathbb{R}` is finite"""

    @abc.abstractmethod
    def jump_of_finite_variation(self) -> bool:
        """:return: true if the integral of :math:`|x| \\nu(dx)` on :math:`|x| <= 1` is finite"""

    def finite_first_moment(self):
        """:return: true if the integral of :math:`|x| \\nu(dx)` on :math:`|x| > 1` is finite"""
        val = quad(func=self.__call__, a=-np.inf, b=-1) + quad(
            func=self.__call__, a=1, b=np.inf
        )
        return val < np.inf

    @abc.abstractmethod
    def blumenthal_getoor_index(self) -> float:
        """:return: the Blumenthal-Getoor index"""

    def x_nu(self, x: float) -> float:
        """:return: the multiplication of x times the Lévy measure

        note:: there are a lot of cases where this expression can be simplified
        """
        return x * self.__call__(x)

    def integrate(self, a: float, b: float) -> float:
        """Integrate the levy measure :math:`\\nu(dx)` between x=a and x=b"""
        if a > b:
            raise ValueError("Expected a<b when integrating the levy measure")
        return quad(lambda x: self.__call__(x), a, b)[0]

    def integrate_against_x(self, a: float, b: float) -> float:
        """Integrate :math:`x \\nu(dx)` between x=a and x=b"""
        if a > b:
            raise ValueError("Expected a<b when integrating the levy measure")
        return quad(self.x_nu, a, b)[0]

    def integrate_against_xx(self, a: float, b: float) -> float:
        """Integrate :math:`x^2 \\nu(dx)` between x=a and x=b"""
        if a > b:
            raise ValueError("Expected a<b when integrating the levy measure")
        return quad(lambda x: x * x * self.__call__(x), a, b)[0]

    def integrate_against_xn(self, a: float, b: float, n: int):
        """Integrate :math:`x^n nu(dx)` between x=a and x=b"""
        if n == 0:
            return self.integrate(a=a, b=a)
        if n == 1:
            return self.integrate_against_x(a=a, b=b)
        if n == 2:
            return self.integrate_against_xx(a=a, b=b)

        if a > b:
            raise ValueError("Expected a<b when integrating the levy measure")
        return quad(lambda x: x**n * self.__call__(x), a, b)[0]


class TruncatedLevyMeasure(LevyMeasure):
    """The truncated Lévy measure is the restriction of a Lévy measure to an interval [a, b]"""

    def __init__(self, levy_measure: LevyMeasure, truncations: tuple[float, float]):
        """
        :param levy_measure: considered Lévy measure
        :param truncations:  truncation parameters a, b with a < b such as the new measure is the restriction to [a, b]
        """
        self.levy_measure = levy_measure
        self.truncations = truncations

    def _truncated_interval(self, a, b):
        """:return: the intersection of the [a,b] with the truncated interval"""
        l, r = self.truncations
        return max(min(a, r), l), min(max(b, l), r)

    def __call__(self, x: np.array) -> np.array:
        l, r = self.truncations
        if x > r or x < l:
            return 0.0
        else:
            return self.levy_measure(x)

    def support(self):
        left, right = self.levy_measure.support()
        return self._truncated_interval(left, right)

    def jump_of_finite_activity(self) -> bool:
        return self.levy_measure.jump_of_finite_activity()

    def jump_of_finite_variation(self) -> bool:
        return self.levy_measure.jump_of_finite_variation()

    def finite_first_moment(self):
        return self.levy_measure.finite_first_moment()

    def blumenthal_getoor_index(self) -> float:
        return self.levy_measure.blumenthal_getoor_index()

    def integrate(self, a: float, b: float) -> float:
        if a > b:
            raise ValueError("Expected a<b when integrating the levy measure")
        aa, bb = self._truncated_interval(a, b)
        return self.levy_measure.integrate(aa, bb)

    def integrate_against_x(self, a: float, b: float) -> float:
        if a > b:
            raise ValueError("Expected a<b when integrating the levy measure")
        aa, bb = self._truncated_interval(a, b)
        return self.levy_measure.integrate_against_x(aa, bb)

    def integrate_against_xx(self, a: float, b: float) -> float:
        if a > b:
            raise ValueError("Expected a<b when integrating the levy measure")
        aa, bb = self._truncated_interval(a, b)
        return self.levy_measure.integrate_against_xx(aa, bb)

    def integrate_against_xn(self, a: float, b: float, n: int):
        if a > b:
            raise ValueError("Expected a<b when integrating the levy measure")
        aa, bb = self._truncated_interval(a, b)
        return self.levy_measure.integrate_against_xn(aa, bb, n)


class LevyTriplet:
    """The Lévy triplet is defined as (a, sigma, nu) where:
    - a is the drift of the given representation
    - sigma is the diffusion coefficient
    - nu is the Lévy measure
    """

    sigma = positive("sigma")

    def __init__(
        self,
        sigma: float,
        nu: LevyMeasure,
        a: float = 0,
        representation: LevyRepresentation = LevyRepresentation.ONEONE,
    ):
        """

        :param sigma: Brownian coefficient
        :param nu: Lévy measure
        :param a: drift term
        :param representation: Lévy-Khintchine representation
        """
        self.a = a
        self.sigma = sigma
        self.nu = nu
        self.representation = representation

        self._drift_mapping = {
            LevyRepresentation.ONEONE: self.canonical_drift,
            LevyRepresentation.TILDE: self.tilde_drift,
            LevyRepresentation.CENTER: self.center_drift,
            LevyRepresentation.ZERO: self.zero_drift,
        }

    def __str__(self):
        return "a={}, sigma={}, nu={}".format(self.a, self.sigma, self.nu)

    def canonical_drift(self) -> float:
        """
        :return: the `canonical` drift, that is the drift corresponding to the cut-off function
                 :math:`c(x) = 1` if :math:`|x| < 1`
        """
        if self.representation not in [
            LevyRepresentation.ZERO,
            LevyRepresentation.CENTER,
            LevyRepresentation.ONEONE,
            LevyRepresentation.TILDE,
        ]:
            raise NotImplementedError("Representation not implemented.")

        if self.representation == LevyRepresentation.ONEONE:  # canonical representation
            return self.a

        adj = 0
        if self.representation == LevyRepresentation.ZERO or (
            self.representation == LevyRepresentation.TILDE
            and self.nu.jump_of_finite_variation()
        ):
            adj = self.nu.integrate_against_x(-1, +1)
        elif self.representation == LevyRepresentation.CENTER:
            adj = -(
                self.nu.integrate_against_x(-np.inf, -1)
                + self.nu.integrate_against_x(+1, np.inf)
            )
        # elif self.representation == LévyRepresentation.TILDE and not finite_integral:
        #     adj = 0

        return self.a + adj

    def zero_drift(self) -> float:
        """
        :return: the drift in the `zero` representation
        """
        canonical_drift = self.canonical_drift()
        adj = -self.nu.integrate_against_x(-1, +1)
        return canonical_drift + adj

    def center_drift(self) -> float:
        """
        :return: centre drift corresponding to the cut-off function :math: `c(x) = 1`
        """
        canonical_drift = self.canonical_drift()
        adj = self.nu.integrate_against_x(-np.inf, -1) + self.nu.integrate_against_x(
            +1, np.inf
        )
        return canonical_drift + adj

    def tilde_drift(self) -> float:
        """
        :return: the drift in the tilde representation
        """
        canonical_drift = self.canonical_drift()
        adj = 0
        if self.nu.jump_of_finite_variation():
            adj = -self.nu.integrate_against_x(-1, +1)
        return canonical_drift + adj

    def set_representation(self, representation: LevyRepresentation) -> None:
        """
        :param representation: Lévy-Khintchine representation
        """
        if representation != self.representation:
            self.a = self._drift_mapping[representation]()
            self.representation = representation  # it's critical to update the representation after the drift is
            # computed as the current representation is used to compute the new drift.
            # another way would be to pass the current representation to the drift calculation function


class Cumulant(abc.ABC):
    """
    Expression of the first 6 cumulants functions of the Lévy model if they exist.
    """

    @abc.abstractmethod
    def __init__(self, drift: float, parameters: Parameters):
        """.. todo:: Enforce init signature in subclasses"""

    def cumulant1(self, t: float) -> float:
        raise NotImplementedError("first cumulant not implemented")

    def cumulant2(self, t: float) -> float:
        raise NotImplementedError("second cumulant not implemented")

    def cumulant3(self, t: float) -> float:
        raise NotImplementedError("third cumulant not implemented")

    def cumulant4(self, t: float) -> float:
        raise NotImplementedError("fourth cumulant not implemented")

    def cumulant5(self, t: float) -> float:
        raise NotImplementedError("fifth cumulant not implemented")

    def cumulant6(self, t: float) -> float:
        raise NotImplementedError("sixth cumulant not implemented")


class LevyModel(Model):
    """This class represents a time-homogeneous Lévy model

    .. note:: the simulated process has the same representation as the model, more explicitly the Lévy model L is
              simulated via the process L (contrary to the :class:`ExponentialOfLévyModel` S = exp(L) which is simulated
              via the log-process L)
    """

    process_representation = ProcessRepresentation.IDENDITY

    def __init__(
        self, model_type: ModelType, levy_triplet: LevyTriplet, cumulant: Cumulant
    ):
        """
        :param model_type: name of the model
        :param levy_triplet: Lévy triplet
        :param cumulant: cumulant expressions
        """
        super().__init__()
        self.model_type = model_type
        self.levy_triplet = levy_triplet
        self.cumulant = cumulant
        self._original_drift = (
            levy_triplet.a
        )  # needed if the Lévy representation is changed

    def __str__(self):
        cls = self.__class__.__name__
        return "{cls} with Lévy triplet: {levy_triplet}".format(
            cls=cls, levy_triplet=str(self.levy_triplet)
        )

    @abc.abstractmethod
    def __repr__(self):
        """repr method"""

    def truncate_levy_measure(self, truncations) -> None:
        """Truncate the Lévy measure: the Lévy measure will return 0 outside the `truncations` interval"""
        truncated_levy_measure = TruncatedLevyMeasure(self.levy_triplet.nu, truncations)
        self.levy_triplet.nu = truncated_levy_measure

    def dimension(self) -> int:
        return 1

    def x0_value(self):
        return 0

    def drift(self, t: float = 0, x: np.array = 0) -> np.array:
        """zero drift, by default, as the process is directly modelled by the Lévy model"""
        return 0

    def process_drift(self) -> np.array:
        return self.levy_triplet.a

    def df(self, t: float) -> float:
        return 1.0

    def diffusion_coefficient(self) -> float:
        """:return: the diffusion coefficient of the stochastic process"""
        return self.levy_triplet.sigma

    def jump_of_finite_activity(self) -> bool:
        return self.levy_triplet.nu.jump_of_finite_activity()

    def jump_of_finite_variation(self) -> bool:
        return self.levy_triplet.nu.jump_of_finite_variation()

    def finite_first_moment(self):
        return self.levy_triplet.nu.finite_first_moment()

    def jump_increment(self, n) -> np.array:
        """Direct simulation of jump increments

        :param n: numbers of jumps to simulate
        :return: array of the jump values
        """
        raise ValueError(
            "jump increments cannot be simulated in closed-form for this model"
        )

    @abc.abstractmethod
    def levy_exponent_pure_jump(self, x: complex) -> complex:
        """The Lévy exponent phi is such that E[exp(x L_t)] = exp(t*phi(x)) where L is the Lévy model
        with triplet (0, 0, nu).

        :return: phi(z) where z = ix, that is phi(x) = levy_exponent_pure_jump(ix)
        """

    def levy_exponent(self, x: complex) -> complex:
        """
        :return: the Lévy exponent phi such that :math:`\\mathbb{E}[exp(u L_t)] = exp(\\phi(u)*t)` where L is the
                 Lévy model with triplet (a, sigma, nu)
        """
        a = self._original_drift
        sigma = self.levy_triplet.sigma
        le = 1j * x * a - 0.5 * (x * sigma) ** 2 + self.levy_exponent_pure_jump(1j * x)
        return le

    def characteristic_function(self, t: float, x: complex) -> complex:
        """:return: the characteristic function of the process L_t, i.e. E[exp(i u L_t)]"""
        return np.exp(t * self.levy_exponent(x))

    def blumenthal_getoor_index(self) -> float:
        return self.levy_triplet.nu.blumenthal_getoor_index()

    def mass(self, a, b, indices: list[int] = None):
        if indices:
            raise NotImplementedError("not yet implemented")
        if isinstance(a, Real):
            return self.levy_triplet.nu.integrate(a=float(a), b=b)
        else:
            return self.levy_triplet.nu.integrate(a=a[0], b=b[0])

    def intensity(self) -> float:
        raise NotImplementedError

    def density(self, t) -> Callable[[float], Callable[[np.array], np.array]]:
        """:return: the density function implied by the COS formula of the exponential of the Lévy model"""
        from ...numerical.cosmethod import COSPricer

        def helper(u):
            return COSPricer(self).density_log(time=t, u=u)

        return helper

    def plot_density(self, t: float, show: bool = False) -> None:
        """:return: plot the density of the Lévy model"""
        fwd = self.x0_value() + self.levy_triplet.a * t
        bins = np.arange(start=-2.0 * fwd, stop=fwd * 2.0, step=0.01)
        y = self.density(t=t)(bins)
        plt.plot(bins, y, "r--", alpha=0.60)
        if show:
            plt.show()

    def cdf(self, t: float, x: np.array):
        """Cumulative distribution function of the exponential of the Lévy process"""
        return np.zeros_like(x)

    def plot_cdf(
        self,
        t: float,
        data: np.array,
        log_normalisation: bool = True,
        show: bool = False,
        title="",
    ) -> None:
        """:return: plot the cumulative distribution function of the Lévy model"""
        fwd = self.x0_value() + self.drift() * t
        bins = np.arange(start=-3.5 * fwd, stop=fwd * 3.5, step=0.01)
        y = self.cdf(t=t, x=bins)
        return LevyModel._helper_plot_cdf(
            fwd, bins, y, log_normalisation, data, title, show
        )

    @staticmethod
    def _helper_plot_cdf(fwd, bins, y, log_normalisation, data, title, show):
        if log_normalisation:
            bins = np.log(bins) - np.log(fwd)
        plt.plot(bins, y, "r", alpha=0.25, label="cdf")

        if data.size:
            if log_normalisation:
                data = np.log(data) - np.log(fwd)

            ecdf = ECDF(data)
            plt.plot(ecdf.x, ecdf.y, "b--", alpha=0.5, label="empirical cdf")

        plt.legend()
        plt.title(title)

        if show:
            plt.show()
