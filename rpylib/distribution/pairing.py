"""Definition of pairing functions and their inverse

A pairing function is a bijection between integers and :math:`N^k`, see https://en.wikipedia.org/wiki/Pairing_function
for more details and also the paper "The Rosenberg-Strong Pairing Function" by Matthew P. Szudzik
(https://arxiv.org/abs/1706.04129)

Here we define extend the pairing functions to Z
"""

from collections import deque
from functools import cache, lru_cache
from itertools import combinations
from math import floor, sqrt

import numpy as np
from sympy import factorint, multiplicity

from ..grid.grid import Coordinates
from ..grid.spatial import CTMCGrid
from ..numerical.numbers import a_n, upper_bound_a_n
from ..tools.generic import lazy_indices_product


class Pairing:
    """Pairing function"""

    def pairing(self, x: tuple[int, ...]) -> int:
        if len(x) > 2:
            return self.pairing2d(self.pairing(x[:-1]), x[-1])
        else:
            return self.pairing2d(*x)

    def projection(self, z: int, dim=2) -> tuple[int, ...]:
        if dim > 2:
            test = self.projection(z, dim-1)
            p, q = test[0], test[1:]
            return self.projection2d(p) + q

        return self.projection2d(z)

    @staticmethod
    def pairing2d(x: int, y: int) -> int:
        pass

    @staticmethod
    def projection2d(z: int) -> tuple[int, int]:
        pass


class Cantor(Pairing):
    """The famous Cantor pairing"""
    def projection(self, z: int, dim=2) -> tuple[int, ...]:
        if dim != 2:
            raise NotImplementedError
        return self.projection2d(z)

    @staticmethod
    def pairing2d(x: int, y: int) -> int:
        # use //2 to force to return int
        return (pow(x + y, 2) + 3*x + y)//2

    @staticmethod
    def projection2d(z: int) -> tuple[int, int]:
        omega = floor((-1 + sqrt(1+8*z))/2)
        return int(z - omega*(omega + 1)/2), int(omega*(omega + 3)/2 - z)


class RosenbergStrong(Pairing):
    _epsilon = 1e-8

    @lru_cache(maxsize=2**5)
    def pairing(self, x: tuple[int, ...]) -> int:
        if (d := len(x)) > 1:
            y = self.pairing(x[:-1])
            m = max(x)
            return y + m**d + (m - x[-1])*((m + 1)**(d-1) - m**(d-1))

        return x[0]

    @lru_cache(maxsize=2**5)
    def projection(self, z: int, dim=2) -> tuple[int, ...]:
        if dim == 1:
            return z,

        # note:: I add epsilon in "m = floor(z**(1/dim) + epsilon)" because of overflow.
        # For example 64**(1/3) gives 3.99999999 which will be rounded to 3 instead of 4
        m = floor(z**(1/dim) + self._epsilon)
        m_d1 = m**(dim-1)
        m_d = m*m_d1
        aux = (m + 1)**(dim-1) - m_d1
        xd = m - floor(max(0, z - m_d - m_d1)/aux)
        p = self.projection(z - m_d - (m - xd)*aux, dim=dim-1)
        return p + (xd,)

    @staticmethod
    def pairing2d(x: int, y: int) -> int:
        sup = max(x, y)
        return sup*(sup + 1) + x - y

    @staticmethod
    def projection2d(z: int) -> tuple[int, int]:
        m = floor(sqrt(z))
        z1 = z - m**2
        if z1 < m:
            return z1, m
        else:
            return m, 2*m - z1


class Szudzik(Pairing):

    @staticmethod
    def pairing2d(x: int, y: int) -> int:
        if x >= y:
            return x**2 + x + y
        else:
            return x + y**2

    @staticmethod
    def projection2d(z: int) -> tuple[int, int]:
        m = floor(sqrt(z))
        z1 = z - m**2
        if z1 < m:
            return z1, m
        else:
            return m, z1 - m


class HyperbolicPairing(Pairing):
    """ From 'Managing storage for extendible arrays' by Rosenberg"""

    @staticmethod
    def pairing2d(x: int, y: int) -> int:
        if x == 0 and y == 0:
            return 0

        xx = x + 1
        yy = y + 1

        n = xx*yy

        z = a_n(n-1)
        n_factorisation = factorint(n)

        if n_factorisation:
            sorted_primes = sorted(n_factorisation)

            offset = 0
            cum_prod_exponents = 1
            for i, prime in enumerate(sorted_primes):
                ri = multiplicity(prime, xx)
                offset += ri*cum_prod_exponents
                cum_prod_exponents *= 1 + n_factorisation[prime]
            z += offset

        return z

    @staticmethod
    @cache
    def projection2d(z: int) -> tuple[int, int]:
        x, y = 0, 0
        if z == 0:
            return x, y

        # find n such that a_n(n-1) <= z
        n = upper_bound_a_n(z)

        # factor n:
        n_factorisation = factorint(n)
        if n_factorisation:
            x_exponents = []
            sorted_primes = sorted(n_factorisation)
            aux = np.array([1 + n_factorisation[key] for key in sorted_primes])
            for i, prime in enumerate(sorted_primes):
                ti = n_factorisation[prime]
                x_exponent = floor((z - a_n(n-1))/np.prod(aux[:i])) % (ti + 1)
                x_exponents.append(x_exponent)

            x = np.prod([prime**r for prime, r in zip(sorted_primes, x_exponents)])
            y = n // x

        return x - 1, y - 1


class PepisKalmar(Pairing):
    # see section 4.2 in https://arxiv.org/pdf/0808.0555.pdf

    @staticmethod
    def _aux_k(z) -> int:
        q, r = divmod(z, 2)
        if r == 0:
            return q
        else:
            return PepisKalmar._aux_k(q)

    @staticmethod
    def _aux_j(z) -> int:
        q, r = divmod(z, 2)
        if r == 0:
            return 0
        else:
            return PepisKalmar._aux_j(q) + 1

    @staticmethod
    def pairing2d(x: int, y: int) -> int:
        return 2**y*(2*x + 1) - 1

    @staticmethod
    def projection2d(z: int) -> tuple[int, int]:
        q, r = divmod(z, 2)
        if r == 0:
            return q, 0
        else:
            x, y = PepisKalmar._aux_k(q), PepisKalmar._aux_j(q) + 1
            return x, y


def mapping_to_z(n: int) -> int:
    """mapping 0, 1, -1, 2, -2, 3, -3,... to 0, 1, 2, 3, 4, 5, 6,..."""
    if n > 0:
        return 2*n - 1
    else:
        return -2*n


def projection_to_z(z: int) -> int:
    """mapping 0, 1, 2, 3, 4, 5, 6,... to 0, 1, -1, 2, -2, 3, -3,..."""
    q, r = divmod(z, 2)
    return q*(2*r - 1) + r


class PairingToZd:
    """
    Pairing :math:`\\mathbb{N}` with :math:`\\mathbb{Z}^d`
    """
    def __init__(self, pairing: Pairing, dimension: int = 2, omit_zero: bool = True):
        self.n_pairing = pairing
        self.dimension = dimension
        self._omitting_zero = 1 if omit_zero else 0

    def project(self, x: int):
        return self.projection(x + self._omitting_zero)

    def pair(self, x: tuple[int, ...]) -> int:
        return self.pairing(x) - self._omitting_zero

    def pairing(self, x: tuple[int, ...]) -> int:
        # project to N^d
        y = tuple(map(mapping_to_z, x))

        # use pairing N^d to N
        z = self.n_pairing.pairing(y)

        return z

    def projection(self, n) -> tuple[int, ...]:
        # project to N^d
        x = self.n_pairing.projection(n, self.dimension)

        # and then to Z for each element
        y = tuple(map(projection_to_z, x))

        return y


class PairingToZ1d:
    """
    Mapping 0, 1, 2, 3,... to the states in an interval :math:`[-L, R], L>0, R>0, L\\neq R`
    """
    def __init__(self, interval: tuple[int, int], omit_zero: bool = True):
        l, r = interval
        self.left, self.right = l, r
        self._omitting_zero = 1 if omit_zero else 0

        if abs(l) < r:
            self._projection = self._projection_with_switch_to_right
        if r < abs(l):
            self._projection = self._projection_with_switch_to_left

        self._switch = False
        self._kk = 0

    @cache
    def project(self, x: int):
        return self._projection(x + self._omitting_zero)

    def pair(self, x: int) -> int:
        if abs(x) <= min(self.right, -self.left):
            paired = mapping_to_z(x)
        else:
            if self.right > - self.left:
                paired = x - self.left
            else:
                paired = self.right - x

        return paired - self._omitting_zero

    @staticmethod
    def _projection(x: int) -> int:
        return projection_to_z(x)

    def _projection_with_switch_to_right(self, x: int) -> int:
        res = projection_to_z(x)
        if self._switch or res < self.left:
            self._switch = True
            self._kk += 1
            val = -self.left + self._kk + 1
            return val
        return res

    def _projection_with_switch_to_left(self, x: int) -> int:
        res = projection_to_z(x)
        if self._switch or res > self.right:
            self._switch = True
            self._kk += 1
            val = -self.right - self._kk
            return val
        return res


class Boundary:
    """Boundary for a domain, the default is that there is no boundary
    """

    def __call__(self, x: np.array) -> bool:
        """return True if x is outside the boundary"""
        return False

    def __repr__(self):
        return 'no boundary'


class RectangleBoundary(Boundary):

    def __init__(self, truncations: list[tuple[int, int]]):
        self.truncations = truncations

    def __call__(self, x: np.array) -> bool:
        # pick the corresponding boundaries
        boundaries = np.zeros_like(x)
        for k, (xi, bounds) in enumerate(zip(x, self.truncations)):
            boundaries[k] = -bounds[0] if xi < 0 else bounds[1]

        return np.any(np.abs(x) > boundaries)

    def __repr__(self):
        return 'rectangle boundary'


class SimplexBoundary(Boundary):

    def __init__(self, truncations: list[tuple[int, int]]):
        self.truncations_left = np.array([t[0] for t in truncations])
        self.truncations_right = np.array([t[1] for t in truncations])

    def __call__(self, x: np.array) -> bool:
        # pick the corresponding boundaries
        normalised_x = x / np.where(x < 0, self.truncations_left, self.truncations_right)
        return np.sum(normalised_x) > 1

    def __repr__(self):
        return 'simplex boundary'


class MyBoundary(Boundary):

    def __init__(self, truncations: list[tuple[int, int]], threshold: float):
        self.truncations = truncations
        self._c = threshold

    def __repr__(self):
        return 'my boundary'

    @staticmethod
    def b_fun(x: float, r1, r2, c):
        if x < c:
            return r2
        if c > r1:
            return 0

        gamma = c/(r1-c)
        alpha = r1*(r2-c)
        beta = (r1-r2)
        return (alpha/x + beta)*gamma

    def __call__(self, x: np.array) -> bool:
        for (xi, xi_boundaries), (xj, xj_boundaries) in combinations(zip(x, self.truncations), 2):
            r1 = -xi_boundaries[0] if xi < 0 else xi_boundaries[1]
            r2 = -xj_boundaries[0] if xj < 0 else xj_boundaries[1]
            if abs(xj) > self.b_fun(abs(xi), r1, r2, self._c):
                return True

        return False


class Domain:
    """
    A domain D is a "regular" volume which meets the following assumptions:
        - it contains the origin
        - there is point :math:`z = (z_1, z_2, ...., z_n)` in D such that :math:`|z_i| > |x_i|` where :math:`x_i` is
          the i-th coordinate in :math:`(0,0,..., 0, x_i, 0, ..., 0)` which is the boundary on the i-th axis of D
        - all hyperplanes are convex
    """

    def __init__(self, boundary: Boundary, grid: CTMCGrid, pairing: PairingToZd):
        self.boundary = boundary
        self.grid = grid
        self.pairing = pairing

    def outside(self, x: Coordinates) -> bool:
        y = np.atleast_1d(x)
        return self.boundary(y)

    def compute_total_number_of_states_and_frontier(self) -> deque[int]:
        # if the grid is 1d, we just return the left and right points
        if self.grid.dimension == 1:
            left_increment = -self.grid.origin_coordinate.value
            right_increment = self.grid.number_of_points() + left_increment - 1
            left_index = self.pairing.pair(left_increment)
            right_index = self.pairing.pair(right_increment)
            res = deque()
            res.appendleft(left_index)
            res.appendleft(right_index)
            return res

        # exhaust all possible states and
        # - count the number of states in the domain
        # - determine the frontier of the domain (i.e. the states that are at the frontier -> in the grid and the domain
        #   but its neighbour states are either outside the grid or outside the domain)
        frontier_state_indices = deque()  # we store the projected state indices, which is less efficient than storing
        # the state increments directly, as we will need to retrieve the state later but more memory efficient
        # than storing all the states (tuples of ints)
        origin_coordinate = self.grid.origin_coordinate
        pairing = self.pairing

        # find the frontier states:
        axes = self.grid.axes
        all_sizes = [axis.size for axis in axes[:-1]]
        last_size = axes[-1].size
        origin_last_coordinate = origin_coordinate[-1]
        left_size, right_size = -origin_last_coordinate, last_size - origin_last_coordinate

        for ks in lazy_indices_product(all_sizes):
            ks_shifted = tuple(ki - origin_last_coordinate for ki in ks)
            outside_states = []
            all_states = []
            for k in range(left_size, right_size):
                state_increment = (ks_shifted + (k,))
                # the state is inside in the grid by construction, so we only test if it is outside the domain
                outside_states.append(self.outside(self.grid[origin_coordinate + state_increment]))
                all_states.append(pairing.pair(state_increment))

            if not all(outside_states):
                frontier_left_index = next(x for x, y in zip(all_states, outside_states) if not y)
                frontier_right_index = next(x for x, y in zip(reversed(all_states), reversed(outside_states)) if not y)
                frontier_state_indices.appendleft(frontier_left_index)
                frontier_state_indices.appendleft(frontier_right_index)
            else:
                axis_state_index = all_states[origin_last_coordinate]
                frontier_state_indices.appendleft(axis_state_index)  # keep the state on the (last) axis

        return frontier_state_indices


class StatesManager:

    def __init__(self, pairing: PairingToZd, domain: Domain, grid: CTMCGrid):
        """
        :param pairing: pairing object
        :param domain: domain of the admissible states
        :param grid: discretisation grid
        """
        frontier_states = domain.compute_total_number_of_states_and_frontier()
        self.frontier_states_indices = frontier_states
        self.max_frontier_indices = max(frontier_states)
        self.domain = domain
        self.origin_coordinates = grid.origin_coordinate
        self.grid = grid
        self.pairing = pairing
        self._last_projected_index = -1

    def is_outside(self, state_increment):
        state = self.origin_coordinates + state_increment
        is_outside_grid = self.grid.outside(state)
        if is_outside_grid:
            return True

        is_outside_domain = self.domain.outside(self.grid[state])
        return is_outside_domain

    def _sample_frontier_state_increment(self):
        index = np.random.choice(self.frontier_states_indices)
        state_increment = self.pairing.project(index)
        return state_increment

    def project_index_to_state_increment(self, x: int, max_logged: int = -1) -> tuple[tuple[int, ...], bool]:
        """Return the state increment and a boolean which is True if we have exhausted all the states in the domain

        :param x: current index to be mapped to a state increment
        :param max_logged: max logged index
        :return: the state increment and a break condition
        """
        is_outside = self.is_outside
        project = self.pairing.project
        if x == max_logged:
            # reset the self._last_projected_index
            self._last_projected_index = -1

        xx = max(x, self._last_projected_index + 1)

        while xx < self.max_frontier_indices:
            if not is_outside(state_increment := project(xx)):
                self._last_projected_index = xx
                return state_increment, False
            xx = xx + 1

        self._last_projected_index = xx
        state_increment = self._sample_frontier_state_increment()
        return state_increment, True
