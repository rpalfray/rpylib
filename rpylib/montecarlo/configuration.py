"""Configuration object for the Monte-Carlo engine.

    :Example:
        - the number of paths
        - the seed for the random generator
        - the use of variance reduction
        - the number of processes to use in parallel (if using the multiprocessing implementation)
        - etc.
"""

import logging
import os
import random
import time
from enum import Enum

import numpy as np

from .multilevel.criteria import ConvergenceCriteria, GilesConvergenceCriteria
from ..product.product import ControlVariates, NoControlVariates, Product


class VarianceReduction(Enum):
    """Variance reduction methods"""
    RICHARDSONEXTRAPOLATION = 1
    ANTITHETIC = 2


class Engine(Enum):
    """Monte-Carlo engine type"""
    STANDARD = 1
    MULTILEVEL = 2


class VarianceReductionMethod:
    """Class wrapper for the variance reduction methods"""
    def __init__(self):
        self._vrm = []

    def add(self, method: VarianceReduction):
        """add variance method"""
        if method not in self._vrm:
            self._vrm.append(method)
        return self

    def has(self, method: VarianceReduction) -> bool:
        """check if method is present"""
        return method in self._vrm


class Configuration:
    """Monte-Carlo engine global configuration"""

    def __init__(self, variance_reduction: VarianceReductionMethod, seed: int = None,
                 control_variates: ControlVariates = None, activate_spot_statistics: bool = False,
                 nb_of_processes: int = None):
        """
        :param variance_reduction: list of variance reduction methods to use
        :param seed: seed for the random generator
        :param control_variates: control variates
        :param activate_spot_statistics: if true then compute the statistics of the modelled underlying spot
        :param nb_of_processes: number of processes used for the parallel processing implementation

            .. note:: if using the multiprocessing implementation, the seed for each process will be chosen randomly as
                      one can't have the same seed for each proces.
        """
        if seed and (nb_of_processes is None or nb_of_processes > 1):
            logging.log(level=logging.WARNING, msg='when using multiprocessing, the random seed is set to a different '
                                                   'value for each process')
        self.seed = seed
        self.activate_spot_statistics = activate_spot_statistics
        self.variance_reduction = variance_reduction or VarianceReductionMethod()
        self.control_variates = control_variates or NoControlVariates()
        self.nb_of_processes = nb_of_processes

    def initialisation_seed(self, multiprocessing: bool = False):
        """Random seed initialisation

        .. note:: if multiprocessing is used, the seed for each process is set randomly as otherwise
                  all the processes would have the same seed
        """
        if self.seed and not multiprocessing:
            np.random.seed(self.seed)
            np.random.default_rng(self.seed)
            random.seed(self.seed)
        else:
            not_deterministic_seed = (os.getpid() * int(time.time())) % 123456789
            np.random.seed(not_deterministic_seed)
            np.random.default_rng(not_deterministic_seed)
            random.seed(not_deterministic_seed)

    def initialisation(self, product: Product) -> None:
        self.control_variates.initialisation(type(product.payoff_underlying))


class ConfigurationStandard(Configuration):
    """Configuration for the standard Monte-Carlo engine"""
    def __init__(self, mc_paths: int = 1_000, variance_reduction: VarianceReductionMethod = None, seed: int = None,
                 control_variates: ControlVariates = None, activate_spot_statistics: bool = False,
                 nb_of_processes: int = None):
        """
        :param mc_paths: number of Monte-Carlo paths
        :param variance_reduction: variance reduction methods
        :param seed: seed of the random generator
        :param control_variates: control variates
        :param activate_spot_statistics: if true, compute the modelled underlying spot
        :param nb_of_processes: number of processes for parallel computing
        """
        super().__init__(variance_reduction=variance_reduction, seed=seed, control_variates=control_variates,
                         activate_spot_statistics=activate_spot_statistics, nb_of_processes=nb_of_processes)
        self.mc_paths = mc_paths


class ConvergenceRates:
    """Definition of the convergence rates (strong, weak and cost) in the MLMC case"""
    def __init__(self, alpha: float = None, beta: float = None, gamma: float = None):
        """
        :param alpha: weak convergence rate
        :param beta: strong convergence rate
        :param gamma: cost convergence rate
            .. note:: the convergence rate are in base 2 in the level `l`, that is in 2**(alpha*l)
        """
        if alpha is not None and beta is not None and gamma is not None:
            if alpha*beta*gamma < 0.0:
                raise ValueError('expected alpha, beta and gamma positive')
            if alpha < 0.5*min(beta, gamma):
                raise ValueError('expected alpha > 0.5 min(beta, gamma)')
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


def compute_convergence_rates(bg_index: float) -> ConvergenceRates:
    """
    :param bg_index: Blumenthal-Getoor index
    :return: convergence rates alpha (weak convergence), beta (strong convergence) and gamma (cost)
    """
    alpha = 1.0 - bg_index/2.0
    beta = 2.0 - bg_index
    gamma = bg_index
    return ConvergenceRates(alpha=alpha, beta=beta, gamma=gamma)


class ConfigurationMultiLevel(Configuration):
    """Configuration for the Multilevel Monte-Carlo engine"""
    def __init__(self, variance_reduction: VarianceReductionMethod = None,
                 convergence_rates: ConvergenceRates = ConvergenceRates(),
                 convergence_criteria: ConvergenceCriteria = None, initial_level: int = 2, maximum_level: int = 50,
                 initial_mc_paths: int = 100, seed: int = None, control_variates: ControlVariates = None,
                 activate_spot_statistics: bool = False, nb_of_processes: int = None):
        """
        :param variance_reduction: variance reduction methods
        :param convergence_rates: convergence rates (weak, strong and cost)
        :param convergence_criteria: convergence criteria of the MLMC algorithm
        :param initial_level: initial level L0
        :param maximum_level: maximum level L
        :param initial_mc_paths: initial number of paths (for any level)
        :param seed: seed of the random generator
        :param control_variates: control variates
        :param activate_spot_statistics: if true, compute the modelled spot underlying
        :param nb_of_processes: number of processes for the multiprocessing implementation
        """
        super().__init__(variance_reduction=variance_reduction, seed=seed, control_variates=control_variates,
                         activate_spot_statistics=activate_spot_statistics, nb_of_processes=nb_of_processes)
        self.convergence_rates = convergence_rates or ConvergenceRates()
        self.convergence_criteria = convergence_criteria or GilesConvergenceCriteria()
        self.initial_level = initial_level
        self.maximum_level = maximum_level
        self.initial_mc_paths = initial_mc_paths
