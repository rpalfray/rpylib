"""
Numpy implement the Poisson random generator as described in Hormann W (1993)
"The transformed rejection method for generating Poisson random variables"
"""

import numpy.random as npr

from ....distribution.sampling import Sampling


class PoissonNumpy(Sampling):
    """Poisson generator from numpy"""

    def __init__(self, lam: float):
        super().__init__()
        self.lam = lam

    def sample(self, size: int = 1):
        self.sampling_cost += size  # constant cost
        return npr.poisson(lam=self.lam, size=size)
