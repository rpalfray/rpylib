"""Generator for standard uniform random variables

Numpy is used here, the generator returns random floats in the half-open interval [0.0, 1.0)
see https://docs.scipy.org/doc/numpy/reference/generated/numpy.random.random_sample.html#numpy.random.random_sample
"""

import numpy as np
import numpy.random as npr

from ..sampling import Sampling


class Uniform(Sampling):
    """Uniform random variate generator"""
    def __init__(self, low: float = 0.0, high: float = 1.0) -> None:
        super().__init__()
        self.low = low
        self.high = high

    def sample(self, size: int = 1) -> np.array:
        self.sampling_cost += size
        return npr.uniform(low=self.low, high=self.high, size=size)
