"""Generator for a Poisson random variable

The numpy generator is used as default
"""

from enum import Enum
from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..sampling import Sampling
from ..univariate.poisson_impl.knuth import Knuth
from ..univariate.poisson_impl.numpyimpl import PoissonNumpy


class ALGORITHM(Enum):
    """Poisson generator algorithms"""

    NUMPY = 1
    KNUTH = 2


class Poisson(Sampling):
    """Poisson random variate"""

    def __init__(self, lam: Union[int, float], algorithm: ALGORITHM = ALGORITHM.NUMPY):
        """
        :param lam: rate/intensity parameter
        :param algorithm: chosen algorithm (numpy by default)

            .. todo:: test Knuth and Hormann algorithms
        """
        super().__init__()
        if algorithm == ALGORITHM.NUMPY:
            self.generator = PoissonNumpy(lam)
        elif algorithm == ALGORITHM.KNUTH:
            self.generator = Knuth(lam)
        else:
            raise ValueError("Poisson algorithm not yet implemented")

    def sample(self, size: int = 1) -> NDArray[float]:
        return self.generator.sample(size=size)

    def cost(self):
        return self.generator.cost()

    def reset_sampling_cost(self):
        self.generator.reset_sampling_cost()
