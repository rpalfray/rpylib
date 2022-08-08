"""TABLE method to generate random variate from discrete probability distribution
"""

import logging
import random
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..sampling import Sampling
from ..variate.alias import AliasMethod


class TableMethod(Sampling):
    """Table method - 32bit implementation"""

    def __init__(self, probabilities: np.array, states: Callable[[list[int]], Any]):
        super().__init__()
        p = np.array(probabilities, dtype=np.float)
        self.states = states
        self.J, self.alias_method = create_table(p, states)
        self._cst = 0.00000000023283064365386963  # =1/2^32

    def sample(self, size: int = 1) -> NDArray[np.float]:
        self.sampling_cost += size
        return np.array(
            [
                _sample_one(self.J, self.alias_method, self._cst, self.states)
                for _ in range(size)
            ]
        )


def _sample_one(
    J, alias_method: AliasMethod, cst: float, states: Callable[[list[int]], Any]
):
    i = random.getrandbits(32)
    ji = J[i & 255]

    if ji >= 0:
        return states(ji)

    return states(alias_method._draw_with_u(i * cst))


def create_table(probabilities, states):
    ks = np.empty(shape=probabilities.size, dtype=np.int16)
    thetas = np.empty_like(probabilities)

    for i, p in enumerate(probabilities):
        aux = 256 * p
        k = int(aux)
        ks[i] = k
        thetas[i] = aux - k

    J = []
    for i in range(probabilities.size):
        J += ks[i] * [i]

    nb_last_elements = abs(256 - len(J))
    J += nb_last_elements * [-1]

    sum_probabilities = sum(thetas)

    if sum_probabilities > 0:
        scaled_probabilities = thetas / sum_probabilities
        alias_method = AliasMethod(scaled_probabilities, states)
        return J, alias_method
    else:
        logging.error("there is 0 probability associated to the current set of states")
        raise ValueError("probabilities is an array of 0s")
