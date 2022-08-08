"""Huffman's Tree method to generate discrete probability distribution
"""

import bisect
from collections.abc import Callable
from operator import attrgetter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..sampling import Sampling
from ..univariate.uniform import Uniform


class HuffmanTree(Sampling):
    def __init__(self, probabilities: np.array, states: Callable[[list[int]], Any]):
        super().__init__()
        self.K = len(probabilities) - 1
        self.head = create_huffman_tree(np.array(probabilities))
        self.uniform = Uniform()
        self.states = states

    def sample(self, size: int = 1) -> NDArray[np.float]:
        us = self.uniform.sample(size=size)
        res_states = np.empty(shape=size, dtype=int)
        head = self.head
        for k in range(size):
            state, sampling_cost = sample_with_u(us[k], head)
            self.sampling_cost += sampling_cost
            res_states[k] = self.states(state)
        return res_states


def sample_with_u(u: float, head: "Node") -> tuple[int, int]:
    sampling_cost = 0
    ptr = head
    while not ptr.is_leaf:
        left = ptr.left_node
        left_val = left.value
        if u < left_val:
            ptr = left
        else:
            u -= left_val
            ptr = ptr.right_node
        sampling_cost += 1

    return ptr.state, sampling_cost


class Node:
    """A node in the tree is defined by its value and whether it is a leaf (or has children nodes)"""

    __slots__ = ("value", "is_leaf")

    def __init__(self, value: float, is_leaf: bool):
        self.value = value
        self.is_leaf = is_leaf


class InternalNode(Node):
    """An internal node is not a lead, that is it has at least a left node child or a right node child"""

    __slots__ = ("left_node", "right_node")

    def __init__(self, value: float, left_node: Node, right_node: Node):
        super().__init__(value=value, is_leaf=False)
        self.left_node = left_node
        self.right_node = right_node


class Leaf(Node):
    """A leaf node has no children"""

    __slots__ = ("state",)

    def __init__(self, value: float, state: int):
        super().__init__(value=value, is_leaf=True)
        self.state = state


class Heap:
    def __init__(self, nodes: [Node]):
        # the heap is sorted in decreasing order (in the value of the nodes)
        self.nodes = nodes
        self.nodes.sort(key=attrgetter("value"), reverse=True)
        self._values = [node.value for node in self.nodes][
            ::-1
        ]  # warning: the values are in increasing order

    def pop(self) -> Node:
        self._values.pop()
        return self.nodes.pop()

    def insert(self, node: Node):
        index = bisect.bisect_left(self._values, node.value)
        self._values.insert(index, node.value)
        index = len(self._values) - index
        self.nodes.insert(index, node)


def create_huffman_tree(probabilities) -> Node:
    length = len(probabilities) - 1
    heap = Heap([Leaf(p, state) for state, p in enumerate(probabilities)])

    for _ in range(length):
        node1 = heap.pop()
        node2 = heap.pop()
        node = InternalNode(node1.value + node2.value, node1, node2)
        heap.insert(node)

    return heap.nodes[0]
