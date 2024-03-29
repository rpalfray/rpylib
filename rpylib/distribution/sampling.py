"""Generic sampling class for continuous or discrete probability distributions
"""


import abc
from enum import Enum

import numpy as np


class SamplingMethod(Enum):
    """Sampling method type"""

    ALIAS = 1
    TABLE = 2
    BINARYSEARCHTREE = 3
    HUFFMANNTREE = 4
    INVERSION = 5
    BINARYSEARCHTREEADAPTED = 6
    BINARYSEARCHTREEADAPTED1D = 7


class Sampling:
    """Base class for sampling method"""

    def __init__(self):
        self.sampling_cost = 0

    def cost(self) -> int:
        """
        :return: the computing cost of the algorithm for generating the random variable
                 the cost will usually correspond to the number of generated uniform random variables
        """
        return self.sampling_cost

    def reset_sampling_cost(self):
        """reset the simulation cost to 0"""
        self.sampling_cost = 0

    @abc.abstractmethod
    def sample(self, size: int = 1) -> np.array:
        """sample a random variable corresponding the distribution in scope

        :param size: size of the sampling vector
        :return: the array of simulated variables
        """
