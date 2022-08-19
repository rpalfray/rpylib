"""ALIAS method to generate random variate from discrete probability distribution

"""

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np

from ..sampling import Sampling
from ..univariate.uniform import Uniform


class AliasMethod(Sampling):
    """Alias Method"""

    def __init__(self, probabilities: np.array, states: Callable[[list[int]], Any]):
        """
        :param probabilities: vector of probabilities (which must sum to 1)
        :param states: discrete spatial states
        """
        super().__init__()
        self.states = states
        p = np.array(probabilities, dtype=float)
        self.K = p.size
        if p.size >= np.iinfo(np.uint).max:
            raise ValueError("AliasMethod: input is too big and will lead to overflow")

        self.J, self.q = create_alias(p)
        self.uniform = Uniform()

    def cost(self):
        return self.uniform.cost()

    def reset_sampling_cost(self):
        return self.uniform.reset_sampling_cost()

    def sample(self, size: int = 1) -> np.array:
        us = self.uniform.sample(size=size)
        gen = [self._draw_with_u(u) for u in us]
        res = self.states(gen)
        return res

    def _draw_with_u(self, uniform: float):
        """ALIAS sampling with pre-generated uniform variable"""
        ku = self.K * uniform
        x = np.uint(ku)
        v = ku - x
        if v < self.q[x]:
            return x
        return self.J[x]


def create_alias(probabilities):
    """Initialisation of the ALIAS method"""
    dim = len(probabilities)
    q = probabilities * dim
    j = np.zeros(shape=dim, dtype=np.uint)

    # sort the scaled probabilities into >1 and <=1
    smaller = deque()
    greater = deque()

    for l in range(dim):
        ql = q[l]
        if ql < 1.0:
            smaller.append(l)
        else:
            greater.append(l)

    while smaller and greater:
        great = greater.pop()
        small = smaller.pop()
        j[small] = great
        q[great] = (q[great] + q[small]) - 1.0

        if q[great] < 1.0:
            smaller.append(great)
        else:
            greater.append(great)

    # this case corresponds to the probabilities ql=1/K
    while greater:
        great = greater.pop()
        q[great] = 1.0

    # this case corresponds to p=1.0 being accidentally converted to 0.999999 and
    # being added into the 'smaller' list instead of the 'greater' list
    while smaller:
        small = smaller.pop()
        q[small] = 1.0

    return j, q
