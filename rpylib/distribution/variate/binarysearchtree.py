"""
Binary Search Tree method to generate discrete probability distribution

"""

from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..sampling import Sampling
from ..univariate.uniform import Uniform


class BinarySearchTree(Sampling):
    def __init__(self, probabilities: np.array, states: Callable[[list[int]], Any]):
        super().__init__()
        self.K = len(probabilities) - 1
        self.bst = create_binary_search_tree(probabilities)
        self.uniform = Uniform()
        self.states = states

    def sample(self, size: int = 1) -> NDArray[np.float]:
        us = self.uniform.sample(size=size)
        return np.array([self.sample_with_u(u) for u in us])

    def sample_with_u(self, u):
        ptr = 1
        while ptr <= self.K:
            if u < self.bst[ptr - 1]:
                ptr = 2 * ptr
            else:
                ptr = 2 * ptr + 1
            self.sampling_cost += 1

        return self.states(ptr - self.K - 1)


def create_binary_search_tree(probabilities):
    k = len(probabilities) - 1
    bst = np.concatenate((np.zeros(shape=k), probabilities))

    ptr = 1
    stack = deque()
    cum_probability = 0

    while True:
        if ptr <= k:
            stack.append(ptr)
            ptr *= 2
        else:
            cum_probability += bst[ptr - 1]
            ptr = stack.pop()
            bst[ptr - 1] = cum_probability
            ptr = 2 * ptr + 1

        if ptr > k and not len(stack):
            break

    return bst[: k + 1]  # no need to return the leaves
