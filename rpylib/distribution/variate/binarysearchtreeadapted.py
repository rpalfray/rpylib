"""Binary Search Tree method. The term `adapted` means two things:
    - the probabilities of the tree are computed on the fly, contrary to the `usual` binary search tree where the
      probabilities are pre-computed.
    - it is adapted to our purpose, that is to the simulation CTMC processes, but the code could be made generic to be
      `adapted` to other simulation algorithms
"""

from functools import lru_cache
from itertools import product

import numpy as np

from rpylib.model.levymodel.levymodel import LevyModel
from ..sampling import Sampling
from ..univariate.uniform import Uniform
from ...grid.grid import Coordinates
from ...grid.spatial import CTMCGrid


class Point:
    """Representation of a point in the state grid"""
    __slots__ = ('coordinates', 'value')

    def __init__(self, coordinates, value):
        """
        :param coordinates: coordinates (x1, x2, x3,...)
        :param value: values (y1, y2, y3, ...)
        """
        self.coordinates = coordinates
        self.value = value

    def __repr__(self):
        return f'Point({self.coordinates}, {self.value})'


class BinarySearchTreeAdapted1D(Sampling):
    def __init__(self, model: LevyModel, grid: CTMCGrid, intensity_of_jumps: float):
        super().__init__()
        self.model = model
        self.axis = grid.axes[0]
        self.uniform = Uniform()

        self.intensity_of_jumps = intensity_of_jumps
        self.origin_coordinate = grid.origin_coordinate.value
        self._proba_left_axis = 0
        if self.intensity_of_jumps > 0:
            self._proba_left_axis = model.mass(-np.inf, -grid.h/2)/self.intensity_of_jumps
        self._coordinates_left_axis = 0, self.origin_coordinate - 1
        self._coordinates_right_axis = self.origin_coordinate+1, len(self.axis) - 1

    @lru_cache(maxsize=2**18)
    def _compute_probability(self, a, b):
        return self.model.mass(a, b)/self.intensity_of_jumps

    def sample(self, size: int = 1) -> np.array:
        res = [self.sample_with_u(u) for u in self.uniform.sample(size=size)]
        return res

    def sample_with_u(self, u: float):
        axis = self.axis
        left, right = self._coordinates_left_axis
        current_p = u
        if u > self._proba_left_axis:
            left, right = self._coordinates_right_axis
            current_p -= self._proba_left_axis

        while left != right:
            middle = (left + right)//2
            l, r = left, middle  # choose left interval by default
            a, b = 0.5*(axis[max(0, l-1)] + axis[l]), 0.5*(axis[r] + axis[min(len(axis) - 1, r+1)])
            p = self._compute_probability(a, b)

            if current_p > p:
                left = min(right, middle+1)
                current_p -= p
            else:
                right = middle

        state_increment = left - self.origin_coordinate
        return state_increment


class BinarySearchTreeAdapted(Sampling):

    def __init__(self, model: LevyModel, grid: CTMCGrid):
        super().__init__()
        ps, cs, is_axis, precomputed_cum_p_for_axes, intensity_of_jumps = self._pre_computation(model=model, grid=grid)
        self._buckets_probabilities = ps
        self._buckets_coordinates = cs
        self._cum_ps = np.cumsum(ps)
        self._is_axis = is_axis
        self._precomputed_cum_p_for_axes = precomputed_cum_p_for_axes

        self.intensity_of_jumps = intensity_of_jumps
        self.grid = grid
        self.model = model
        self.uniform = Uniform(high=sum(ps))

    @staticmethod
    def _pre_computation(model: LevyModel, grid: CTMCGrid):
        origin_coordinate = grid.origin_coordinate
        h_left = grid.middle(grid.left_point(grid.origin_coordinate), grid.origin)
        h_right = grid.middle(grid.origin, grid.right_point(grid.origin_coordinate))

        # break each axis into [-R, -h/2] x ]-h/2, h/2[ x [h/2, R]
        # and compute the coordinates for these 3 pieces for each axis
        # we sort them as ]-h/2, h/2[, [-R, -h/2], [h/2, R] see the next comment
        intervals = []
        for k, axis in enumerate(grid.axes):
            axis_origin_c = origin_coordinate[k]
            hl, hr = h_left[k], h_right[k]
            cs = [axis_origin_c, axis_origin_c], [0, axis_origin_c-1], [axis_origin_c+1, len(axis)-1]
            vs = (hl, hr), (axis[0], hl), (hr, axis[-1])
            pts = tuple((Point(cl, vl), Point(cr, vr)) for (cl, cr), (vl, vr) in zip(cs, vs))
            intervals.append(pts)

        cartesian_product = product(*intervals)
        # discard first set which is [h_l1, h_r1]x[h_l2, h_r2]x...x[h_ln, h_rn]
        # we are calculating the measure of all the sets on the complement of this very set
        next(cartesian_product)
        intensity_of_jumps = 0
        masses = []
        buckets_coordinates = []
        is_cached_axis = []
        low_nb_of_pts = len(grid.axes)*len(grid.axes[0]) < 10_001
        for c_set in cartesian_product:
            a_pts, b_pts = zip(*c_set)
            a = [a_pt.value for a_pt in a_pts]
            b = [b_pt.value for b_pt in b_pts]
            a_c = [a_pt.coordinates for a_pt in a_pts]
            b_c = [b_pt.coordinates for b_pt in b_pts]
            p = model.mass(a, b)
            intensity_of_jumps += p
            is_an_axis_and_low_nb_pts = low_nb_of_pts and sum(l != r for l, r in zip(a_c, b_c)) == 1
            masses.append(p)
            buckets_coordinates.append(list(zip(a_c, b_c)))
            is_cached_axis.append(is_an_axis_and_low_nb_pts)

        precomputed_cum_p_for_axes = {}
        for k, is_an_axis in enumerate(is_cached_axis):
            if is_an_axis:
                a_c, b_c = list(zip(*buckets_coordinates[k]))
                axis_nb = next(k for k, (l, r) in enumerate(zip(a_c, b_c)) if l != r)
                ps = []
                pt_list = list(a_c)
                for c in range(a_c[axis_nb], b_c[axis_nb] + 1):
                    pt_list[axis_nb] = c
                    pt = Coordinates(pt_list)
                    left_middle_point = grid.middle(grid.left_point(pt), grid[pt])
                    right_middle_point = grid.middle(grid[pt], grid.right_point(pt))
                    p = model.mass(a=left_middle_point, b=right_middle_point)/intensity_of_jumps
                    ps.append(p)
                precomputed_cum_p_for_axes[k] = np.cumsum(ps)

        probabilities = [mass/intensity_of_jumps for mass in masses]

        return probabilities, buckets_coordinates, is_cached_axis, precomputed_cum_p_for_axes, intensity_of_jumps

    @lru_cache(maxsize=2**18)
    def _compute_probability(self, a, b):
        return self.model.mass(a, b)/self.intensity_of_jumps

    def cost(self) -> int:
        return self.uniform.cost()

    def reset_sampling_cost(self) -> None:
        self.uniform.reset_sampling_cost()

    def sample(self, size: int = 1) -> np.array:
        res = self.sample_with_us(self.uniform.sample(size=size))
        return res

    def sample_with_us(self, us: np.array):
        # find the bucket where to sample the state
        bucket_positions = np.searchsorted(self._cum_ps, us)
        bucket_coordinates = [self._buckets_coordinates[bucket_position] for bucket_position in bucket_positions]
        positions = np.where(bucket_positions > 0)
        us[positions] -= np.array([self._cum_ps[bucket_positions[position]-1] for position in positions]).ravel()

        state_increments = []
        for these_bucket_coordinates, bucket_position, prob in zip(bucket_coordinates, bucket_positions, us):
            if self._is_axis[bucket_position]:
                # find the position of the state
                state_ith_pos = np.searchsorted(self._precomputed_cum_p_for_axes[bucket_position], prob)
                a_c, b_c = list(zip(*these_bucket_coordinates))
                state = tuple(l if l == r else l + state_ith_pos for l, r in zip(a_c, b_c))
                state_increment = tuple(v - o for v, o in zip(state, self.grid.origin_coordinate))
            else:
                state_increment = self.sample_one_bucket(coordinates=these_bucket_coordinates, probability=prob)
            state_increments.append(state_increment)

        return state_increments

    def sample_one_bucket(self, coordinates, probability: float):
        result = list(coordinates)
        grid = self.grid
        axes = grid.axes
        current_probability = probability

        while any(l != r for l, r in result):
            # iterate the binary search successively on each axis
            for k, ((left, right), axis) in enumerate(zip(result, axes)):
                if left != right:
                    middle = (right + left) // 2
                    result[k] = left, middle
                    a_c, b_c = zip(*result)
                    a_cc = Coordinates(a_c)
                    b_cc = Coordinates(b_c)
                    a = grid.middle(grid.left_point(a_cc), grid[a_cc])
                    b = grid.middle(grid[b_cc], grid.right_point(b_cc))
                    p = self._compute_probability(a, b)
                    if current_probability > p:
                        result[k] = min(right, middle+1), right
                        current_probability -= p

        state_increment = tuple(c[0] - o for c, o in zip(result, grid.origin_coordinate))
        return state_increment
