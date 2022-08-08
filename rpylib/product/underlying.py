"""Definition of a financial underlying.

   :Example:
        - Spot underlying:  the most common underlying corresponding to the spot price
        - Asian underlying: average of the considered underlying over a period of time
        - Libor underlyings: libor-like underlyings

"""

import abc
import copy
from collections.abc import Callable
from enum import Enum
from math import exp

import numpy as np

from ..grid.time import TimeGrid
from ..process.process import ProcessRepresentation


class UnderlyingDimension(Enum):
    """Dimension of the underlying; it is multidimensional if the payoff is a function of more than one underlying.

    :Example:
        - a standard equity Call option is unidimensional
        - a Rainbow option is `multidimensional` as it involves to compute the maximum of performances
    """

    ONEDIMENSIONAL = 1
    MULTIDIMENSIONAL = 2


class Discretisation(Enum):
    """Discretisation type for some non-trivial underlying

    :Example:
        - Asian option can be daily/weekly/etc. average
    """

    DAILY = 1
    WEEKLY = 2
    MONTHLY = 3
    YEARLY = 4


def discretisation_year_fraction(discretisation: Discretisation) -> float:
    """
    Convert the discretisation enum to the corresponding year fraction
    By default, the year fraction convention is Act365.

    :param discretisation: discretisation type
    :return: the year fraction
    """
    if discretisation == Discretisation.DAILY:
        return 1.0 / 365.0
    if discretisation == Discretisation.WEEKLY:
        return 1.0 / 52.0
    if discretisation == Discretisation.MONTHLY:
        return 1.0 / 12.0
    if discretisation == Discretisation.YEARLY:
        return 1.0
    raise NotImplementedError("discretisation type is not yet handled")


class Underlying(abc.ABC):
    """Abstract class for an underlying object

    .. note:: the values of the process might be passed as the logarithms of the spot for optimisation purpose.
    """

    underlying_dimension = UnderlyingDimension.ONEDIMENSIONAL

    @abc.abstractmethod
    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        """call method when the process is simulated under the same representation

        :param times: :math:`t_0, t_1,..., t_n`
        :param path: :math:`log(S_0), log(S_1),...,log(S_n)`
        :param jump_path: :math:`log(J_0), log(J_1),..., log(J_n)` where :math:`J_i` is the jump at time :math:`t_i`
                     some payoffs need the fine structure of the jumps (for example the DefaultTime underlying)
        :param payoff_underlying: payoff underlying valued passed for optimisation purpose
        :return: value of the underlying for the trajectory defined by (times, path)
        """

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        """Underlying value from the process X simulated under the representation log(X)"""
        raise NotImplementedError(
            "Process log-representation not implemented for underlying "
            + self.__class__.__name__
        )

    def update(self, process_representation: ProcessRepresentation):
        """update the underlying given the process representation type"""
        if process_representation == ProcessRepresentation.LOG:
            self.value = self._value_log

    def imply_from_payoff_underlying(self, payoff_underlying_type) -> Callable:
        """
        If the underlying is closely related (potentially the same) to the payoff underlying, then we can use this
        knowledge to speed up the computation of the underlying in scope.

        :param payoff_underlying_type: underlying type of the payoff underlying
        :return: a function that will take the times, the path and the payoff underlying value and return the
                 underlying value from the relevant quantities
        """
        if isinstance(self, payoff_underlying_type):
            return lambda times, path, jump_path, payoff_underlying: payoff_underlying

        return self.value

    def check_consistency(self, process_dimension: int):
        if (self.underlying_dimension == UnderlyingDimension.MULTIDIMENSIONAL) and (
            process_dimension > 1
        ):
            pass
            # FIXME: a rainbow option needs a multidimensional underlying in which case we don't want to throw
            #  the following error:
            #
            # raise ValueError('The process is multidimensional and this is not consistent with the payoff underlying,'
            #                  'the payoff underlying must be a function resulting in a real value
            #                  (from example an Asian underlying) not a vector
            #                  (as would be given for a multidimensional Spot underlying).')

    def compute_times_grid(self, maturity: float) -> TimeGrid:
        """
        Compute the time axes adapted to the underlying:
        - a spot underlying will only return 0, maturity
        - an Asian underlying will return t_0, t_1,..., t_n where the t_i are the times when the underlying is averaged

        :param maturity: maturity of the product
        :return: the time axes' discretisation adapted to the underlying
        """
        return TimeGrid(start=0.0, end=maturity)


class Spot(Underlying):
    """Spot underlying, the standard underlying used in financial payoffs. This is the spot at maturity."""

    underlying_dimension = UnderlyingDimension.MULTIDIMENSIONAL

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        return path[..., -1]

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        """:return: the spot value corresponding to the last simulated time"""
        return np.exp(path[..., -1])


# Libors is just an alias for Spot (with multiple underlying) but the underlying dimension is set to 1
class Libors(Underlying):
    """Libors underlyings.

    .. note:: this object represents a Libor-like underlying and can used in the Lévy Forward Market too.
    """

    underlying_dimension = UnderlyingDimension.ONEDIMENSIONAL

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        return path[..., -1]

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        """:return: the spot value corresponding to the last simulated time"""
        return np.exp(path[..., -1])


class LogSpot(Underlying):
    """Log-Spot underlying, simply the logarithm of the spot underlying"""

    underlying_dimension = UnderlyingDimension.MULTIDIMENSIONAL

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        """:return: the logarithm of the last spot underlying"""
        return np.log(path[..., -1])

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        """:return: the logarithm of the last spot underlying"""
        return path[..., -1]


class Asian(Underlying):
    """Arithmetic average of a single underlying"""

    underlying_dimension = UnderlyingDimension.MULTIDIMENSIONAL

    def __init__(self, discretisation: Discretisation = Discretisation.DAILY):
        """Arithmetic Asian underlying"""
        self.yf = discretisation_year_fraction(discretisation)
        self._spot = Spot()

    def update(self, process_representation: ProcessRepresentation):
        self._spot.update(process_representation)

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        """:return: the average of the spot underlying over the times"""
        res, last_t = 0, 0
        path = self._spot.value(times, path, jump_path, payoff_underlying)
        for t, val in zip(times, path):
            last_t, res = t, res + val * (t - last_t)

        return res / last_t

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        return self.value(times, np.exp(path), np.exp(jump_path), payoff_underlying)

    def compute_times_grid(self, maturity: float) -> TimeGrid:
        num = int(maturity / self.yf) + 1
        if num < 2:
            raise ValueError(
                "The maturity of the product is too small and inconsistent with the underlying"
            )
        return TimeGrid(start=0.0, end=maturity, num=num)


class Mean(Underlying):
    """Arithmetic average of several underlyings"""

    def __init__(self):
        self._spot = Spot()

    def update(self, process_representation: ProcessRepresentation):
        self._spot.update(process_representation)

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        spots = self._spot.value(times, path, jump_path, payoff_underlying)
        return np.mean(spots)

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        return self.value(times, np.exp(path), np.exp(jump_path), payoff_underlying)


class Performances(Underlying):
    """Performances vector of underlyings, that is the ratios of the spot underlyings
    at maturity by their initial values"""

    underlying_dimension = UnderlyingDimension.MULTIDIMENSIONAL

    def __init__(self, spots: list[float]):
        self.spots = np.array(spots)
        self.log_spots = np.log(spots)

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> float:
        return path[..., -1] / self.spots

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        log_performances = path[..., -1] - self.log_spots
        return np.exp(log_performances)


class MaximumOfPerformances(Underlying):
    """Maximum of performances of underlyings"""

    def __init__(self, spots: list[float]):
        """
        :param spots: initial spots values

            .. note:: the performance is the ratios of the underlyings at time T over their initial values at time 0
        """
        self.spots = np.array(spots)
        self.log_spots = np.log(spots)

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> float:
        performances = path[..., -1] / self.spots
        return max(performances)

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        log_performances = path[..., -1] - self.log_spots
        return exp(max(log_performances))


class NthSpot(Underlying):
    """Value of the spot of the n-th underlying among M underlyings (M>=n)"""

    def __init__(self, index: int):
        """
        :param index: underlying index, index=1 corresponds to the first spot S1.
        """
        self.index = index
        if index == 0:
            raise ValueError(
                "expected index > 0, index=k means this is the k-th underlying spot"
            )

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        """:return: the logarithm of the last spot underlying"""
        return path[self.index - 1, -1]

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        """:return: the logarithm of the last spot underlying"""
        return np.exp(path[self.index - 1, -1])

    def imply_from_payoff_underlying(self, payoff_underlying_type) -> Callable:
        if payoff_underlying_type is Spot:
            return lambda times, path, payoff_underlying: payoff_underlying[
                self.index - 1
            ]

        return super().imply_from_payoff_underlying(payoff_underlying_type)


class Indicators(Underlying):
    """Indicator functions, that is, it is equal to 1 if above the threshold else 0"""

    underlying_dimension = UnderlyingDimension.MULTIDIMENSIONAL

    def __init__(self, thresholds: list[float]):
        """
        For the moment, this is the product of indicators with > condition
        :param thresholds: the indicator function is equal to 1 if greater than the threshold, 0 otherwise
        """
        self.thresholds = thresholds
        self.log_thresholds = np.log(thresholds)

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        res = 1 if np.all(path[..., -1] > self.thresholds) else 0
        return np.array([res])

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        res = 1 if np.all(path[..., -1] > self.log_thresholds) else 0
        return np.array([res])


class DefaultTime(Underlying):
    """
    Default times underlyings as modelled in the paper 'A Structural Jump Threshold Framework for Credit Risk'
    by Garreau and Kercheval
    """

    def __init__(self, default_level: float):
        if default_level >= 0:
            raise ValueError("Expected strictly negative default level")
        self._a = default_level

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        log_jump_path = np.log(jump_path)
        return self._value_log(times, path, log_jump_path, payoff_underlying)

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        log_jump_ratio = np.diff(jump_path)
        default_time = np.inf
        idx = np.argwhere(log_jump_ratio < self._a)
        if idx.size > 0:
            default_time = times[np.min(idx) + 1]

        return default_time


class _DefaultTimes(Underlying):
    """
    Default times underlyings as modelled in the paper 'A Structural Jump Threshold Framework for Credit Risk'
    by Garreau and Kercheval.

    .. seealso:: :class:`DefaultTime` but here this is for a multidimensional model and therefore modelling
    the corresponding default times.
    """

    underlying_dimension = UnderlyingDimension.MULTIDIMENSIONAL

    def __init__(self, default_levels: list[float]):
        if any(a >= 0 for a in default_levels):
            raise ValueError("Expected strictly negative default levels")
        self._a = np.array(default_levels)
        self._default_times_inf = np.full(shape=len(default_levels), fill_value=np.inf)

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        log_jump_path = np.log(jump_path)
        return self._value_log(times, path, log_jump_path, payoff_underlying)

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        log_ratios = np.diff(jump_path)
        default_times = copy.copy(self._default_times_inf)
        for k, (log_ratio, a) in enumerate(zip(log_ratios, self._a)):
            idx = np.argwhere(log_ratio < a)
            if idx.size > 0:
                default_times[k] = times[np.min(idx) + 1]

        return np.array(default_times)


class DefaultTimeNthUnderlying(_DefaultTimes):
    """Default time of the n-th underlying among M underlying (M>=n)"""

    underlying_dimension = UnderlyingDimension.ONEDIMENSIONAL

    def __init__(self, default_levels: list[float], underlying_index: int):
        super().__init__(default_levels=default_levels)
        if underlying_index == 0:
            raise ValueError(
                "expected index > 0, index=k means this is the k-th underlying spot"
            )
        self._k = underlying_index - 1
        self._a = default_levels[underlying_index - 1]

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        log_jump_ratio = np.diff(
            np.log(jump_path[self._k, ...])
        )  # consider just the k-th underlying
        default_time = np.inf
        idx = np.argwhere(log_jump_ratio < self._a)
        if idx.size > 0:
            default_time = times[np.min(idx) + 1]

        return default_time

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        log_jump_ratio = np.diff(
            jump_path[self._k, ...]
        )  # consider just the k-th underlying
        default_time = np.inf
        idx = np.argwhere(log_jump_ratio < self._a)
        if idx.size > 0:
            default_time = times[np.min(idx) + 1]

        return default_time


class NthDefaultTimes(_DefaultTimes):
    """N-th default times, that is the first time when at least n underlyings (out of M, M>n) have defaulted"""

    underlying_dimension = UnderlyingDimension.ONEDIMENSIONAL

    def __init__(self, default_levels: list[float], index: int):
        """
        :param index: underlying index, index=1 corresponds to the first spot S1.
        """
        super().__init__(default_levels=default_levels)
        if index == 0:
            raise ValueError(
                "expected index > 0, index=k means this is the k-th default times"
            )
        self._k = index - 1

    def value(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        default_times = super().value(times, path, jump_path, payoff_underlying)
        index_smallest = np.argpartition(default_times, self._k)[: self._k + 1]
        default_time = np.amax(default_times[index_smallest])
        return default_time

    def _value_log(
        self, times, path: np.array, jump_path: np.array, payoff_underlying=None
    ) -> np.array:
        default_times = super()._value_log(times, path, jump_path, payoff_underlying)
        index_smallest = np.argpartition(default_times, self._k)[: self._k + 1]
        default_time = np.amax(default_times[index_smallest])
        return default_time
