"""Defines the simulated process for a Lévy model

The natural case is to simulate a model from its definition, that is by simulating its underlying variables;
nevertheless, this is generally not possible.
One can also approximate the jump part by a discrete Continuous-Time Markov Chain (CTMC) and simulate this chain via
Monte-Carlo methods.
"""

import abc
from enum import Enum

import numpy as np

from ..model.model import Model


class ProcessRepresentation(Enum):
    IDENDITY = 1  # the process X is simulated via the same representation X
    LOG = 2       # the process X is simulated via the representation Y = log(X)


class Process(abc.ABC):

    def __init__(self, model: Model, process_representation: ProcessRepresentation):
        self.model = model
        self.process_representation = process_representation

    def initialisation(self, product: 'Product', max_step_epsilon: float = None) -> None:
        """Initialisation of the process from the characteristics of the product in scope

        :param product: financial product
        :param max_step_epsilon: if epsilon != None then jump times with max step epsilon are used to build the time
                                 discretisation steps
        """
        pass

    def dimension(self) -> int:
        """Number of modelled underlyings"""
        return self.model.dimension()

    def process_drift(self) -> np.array:
        """Drift of the simulated process"""
        return self.model.process_drift()

    def deterministic_path(self, times: np.array) -> np.array:
        """
        :param times: times of the path
        :return: the deterministic part of the path
        """
        return self.model.x0_value() + self.process_drift()*times

    @abc.abstractmethod
    def one_simulation_cost(self, product) -> float:
        """ Returns: the simulating cost corresponding to the numbers of uniform random variables simulated"""

    @abc.abstractmethod
    def reset_one_simulation_cost(self) -> None:
        """ Reset the simulation cost"""

    @abc.abstractmethod
    def pre_computation(self, mc_paths: int,  product: 'Product') -> None:
        """Pre-computation, for example simulate the random variables if possible
        :param mc_paths: number of Monte-Carlo paths
        :param product: financial product to price
        """

    @abc.abstractmethod
    def simulate_one_path(self) -> 'StochasticPath':
        """Simulate (only) one path of the non-deterministic part of the underlying"""

    def df(self, t: float) -> float:
        """Discount factor function

        :param t: maturity
        :return: the discount factor at time t
        """
        return self.model.df(t)
