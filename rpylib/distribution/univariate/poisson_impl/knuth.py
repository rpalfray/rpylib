"""
Simple acceptance-rejectance algorithm for generating Poisson random variable by Donald Knuth

see https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
"""

import numpy as np

from ..uniform import Uniform
from ...sampling import Sampling


class Knuth(Sampling):
    """Knuth algorithm for generating Poisson random variable"""
    def __init__(self, lam: int):
        super().__init__()
        self.lam = lam
        self._L = np.exp(-lam)
        self._U = Uniform()

    def cost(self):
        return self._U.cost

    def reset_sampling_cost(self):
        self._U.reset_sampling_cost()

    def sample_one(self):
        k = 0
        p = 1
        sampling_u = self._U.sample

        while p > self._L:
            k += 1
            p *= sampling_u()

        return k - 1

    def sample(self, size: int = 1):
        res = np.array([self.sample_one() for _ in range(size)])
        return res
