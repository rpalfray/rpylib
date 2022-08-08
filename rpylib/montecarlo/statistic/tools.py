"""Useful tool for computing statistics

"""

from functools import cached_property

import numpy as np
import scipy.stats
import scipy.stats


def mean(simulations: np.array) -> np.array:
    if simulations.shape[0] == 0:  # nothing to do here
        return np.zeros(shape=simulations.shape[1:])
    return np.mean(simulations, axis=0)


def stddev(simulations: np.array) -> np.array:
    if simulations.shape[0] == 1:  # only one piece of data hence no stddev
        return 0.0
    return np.std(
        simulations, axis=0, ddof=1
    )  # ddof=1 to have an unbiased estimator of the variance


def mc_stddev(simulations: np.array) -> np.array:
    if simulations.size == 0:
        return 0.0
    return stddev(simulations) / np.sqrt(simulations.size)


def skewness(simulations: np.array) -> np.array:
    return scipy.stats.skew(simulations, axis=0)


def kurtosis(simulations: np.array) -> np.array:
    return scipy.stats.kurtosis(simulations, axis=0, fisher=False, bias=False)


class NonCenteredMoments:
    """Computation of the moments E[X^k] of X"""

    def __init__(self, simulations: list[np.array]):
        """
        :param simulations: list of ndarray, each element corresponds to the simulations of the level l of
        the Multilevel Monte-Carlo (only one element for a standard Monte-Carlo engine)
        """
        self.simulations = simulations
        self._moments = {}

    @cached_property
    def ncm_first(self):
        """
        :return: first non-centered moment, ie E[X]
        """
        m1_nc = np.array([np.mean(samples) for samples in self.simulations])
        self._moments["m1_c"] = np.zeros(shape=m1_nc.shape[0])
        self._moments["m1_nc"] = m1_nc
        return m1_nc

    @cached_property
    def ncm_second(self):
        """
        :return: second non-centered moment, ie E[X^2]
        """
        if "m1_nc" not in self._moments:
            _ = self.ncm_first
        mu = self._moments["m1_nc"]
        m2_c = np.array(
            [scipy.stats.moment(samples, moment=2) for samples in self.simulations]
        )
        m2_nc = m2_c + mu**2
        self._moments["m2_c"] = m2_c
        self._moments["m2_nc"] = m2_nc
        return m2_nc

    @cached_property
    def ncm_third(self):
        """
        :return: third non-centered moment, ie E[X^3]
        """
        if "m1_nc" not in self._moments:
            _ = self.ncm_first
        mu = self._moments["m1_nc"]
        if "m2_c" not in self._moments:
            self.ncm_second()
        m2_c = self._moments["m2_c"]
        m3_c = np.array(
            [scipy.stats.moment(samples, moment=3) for samples in self.simulations]
        )
        m3_nc = m3_c + mu**3 + 3 * mu * m2_c
        self._moments["m3_c"] = m3_c
        self._moments["m3_nc"] = m3_nc
        return m3_nc

    @cached_property
    def ncm_fourth(self):
        """
        :return: fourth non-centered moment, ie E[X^4]
        """
        if "m1_nc" not in self._moments:
            _ = self.ncm_first
        mu = self._moments["m1_nc"]
        if "m2_c" not in self._moments:
            self.ncm_second()
        if "m2_c" not in self._moments:
            self.ncm_third()
        m2_c = self._moments["m2_c"]
        m3_c = self._moments["m3_c"]
        m4_c = np.array(
            [scipy.stats.moment(samples, moment=4) for samples in self.simulations]
        )
        m4_nc = m4_c + mu**4 + 4 * mu * m3_c + 6 * mu**2 * m2_c
        self._moments["m4_c"] = m4_c
        self._moments["m4_nc"] = m4_nc
        return m4_nc
