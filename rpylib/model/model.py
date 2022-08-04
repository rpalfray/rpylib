"""Generic class for a pricing model
"""

import abc
from enum import Enum

import numpy as np


class ModelType(Enum):
    BLACKSCHOLES = 1    # Black-Scholes model
    MERTON = 2          # Merton model
    HEM = 3             # Hyper-Exponential Jump Diffusion model
    VG = 4              # Variance-Gamma model
    CGMY = 5            # Car-Geman-Madam-Yor model


class Parameters(abc.ABC):
    """Wrapper defining parameters class, the only function here is :func:`initialisation`
       which is needed for the calibration process.
    """

    def initialisation(self):
        """When parameters are calibrated, this function updates dependent members of the Parameter class"""
        pass


class Model:
    """Abstract class of a pricing model, subclasses need to implement a few functions::
        :func:`dimension`
        :func:`drift`
        :func:`process_drift` if the model can be directly simulated
        :func:`df`
    """
    def __init__(self):
        # nothing to see here, move along
        pass

    @abc.abstractmethod
    def dimension(self) -> int:
        """number of modelled underlyings"""

    def dimension_model(self) -> int:
        """dimension of the model, that is the number of factors or drivers in the model"""
        return self.dimension()

    @abc.abstractmethod
    def drift(self, t: float = 0, x: np.array = 0) -> np.array:
        """Drift mu(t, x) of the stochastic process. Most of the time it is a constant drift in time
        and in the underlying variable x.
        :param t: time t
        :param x: value at time t of the underlyings
        """

    def process_drift(self) -> np.array:
        """Drift mu(t, x) of the underlying stochastic process
        """
        raise NotImplementedError('This model cannot be simulated directly or it has not been implemented yet')

    @abc.abstractmethod
    def df(self, t: float) -> float:
        """Discount factor function
        :param t: time t
        :return: the discount factor at time t
        """
