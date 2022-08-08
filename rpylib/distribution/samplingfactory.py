"""Simple factory function to create a sampling method"""

from itertools import product
from typing import Union

import numpy as np

from rpylib.model.levymodel.levymodel import LevyMeasure, LevyModel
from .pairing import (
    PairingToZd,
    Szudzik,
    RosenbergStrong,
    StatesManager,
    Domain,
    Boundary,
    PairingToZ1d,
)
from .sampling import SamplingMethod
from .variate.alias import AliasMethod
from .variate.binarysearchtree import BinarySearchTree
from .variate.binarysearchtreeadapted import (
    BinarySearchTreeAdapted,
    BinarySearchTreeAdapted1D,
)
from .variate.huffmantree import HuffmanTree
from .variate.inversion import InversionMethod
from .variate.table import TableMethod
from ..grid.spatial import CTMCGrid
from ..model.levycopulamodel import LevyCopulaModel


def compute_intensity_of_jumps(
    model: Union[LevyModel, LevyCopulaModel], grid: CTMCGrid
):
    h_left = grid.middle(grid.left_point(grid.origin_coordinate), grid.origin)
    h_right = grid.middle(grid.origin, grid.right_point(grid.origin_coordinate))

    if model.dimension_model() == 1:
        h_left = [h_left]
        h_right = [h_right]

    intervals = [
        [[h_l, h_r], [grid.axes[k][0], h_l], [h_r, grid.axes[k][-1]]]
        for k, (h_l, h_r) in enumerate(zip(h_left, h_right))
    ]
    cartesian_product = product(*intervals)
    # discard first set which is [h_l1, h_r1]x[h_l2, h_r2]x...x[h_ln, h_rn]
    # we are calculating the measure on the complement of this very set
    next(cartesian_product)

    intensity_of_jumps = 0
    for c_set in cartesian_product:
        a, b = zip(*c_set)
        intensity_of_jumps += model.mass(a=a, b=b)

    return intensity_of_jumps


def create_q_vector(levy_measure: LevyMeasure, grid: CTMCGrid) -> np.array:
    int_lm = levy_measure.integrate
    m_middle = grid.origin_coordinate
    q = np.zeros(len(grid.axes[0]), dtype=np.float)

    for k, x in enumerate(grid.axes[0]):
        if k != m_middle:
            x_left = grid.middle(grid.left_point(k), x)
            x_right = grid.middle(x, grid.right_point(k))
            q[k] = int_lm(x_left, x_right)

    return q


def create_vec_jump_matrix(q_vector: np.array, init_state, intensity_of_jumps: float):
    res = q_vector / intensity_of_jumps
    res[init_state.value] = 0.0
    return res


def create_matrix_jump_matrix(q_matrix: np.array, init_state) -> np.array:
    qi = -q_matrix[init_state]
    res = q_matrix / qi
    res[init_state] = 0.0
    return res


def create_sampling_method(
    model,
    levy_measure,
    method: SamplingMethod,
    grid: CTMCGrid,
    is_levy_copula: bool,
    intensity_of_jumps: float,
):
    if method == SamplingMethod.INVERSION:
        return create_sampling_inversion_method(
            grid, model, intensity_of_jumps, is_levy_copula
        )
    if method == SamplingMethod.BINARYSEARCHTREEADAPTED1D:
        return BinarySearchTreeAdapted1D(
            model=model, grid=grid, intensity_of_jumps=intensity_of_jumps
        )
    if not is_levy_copula:
        init_state = grid.origin_coordinate
        q_vector = create_q_vector(levy_measure, grid)
        jump_vector = create_vec_jump_matrix(
            q_vector=q_vector,
            init_state=init_state,
            intensity_of_jumps=intensity_of_jumps,
        )
        left, right = levy_measure.support()
        pivot = grid.origin_coordinate
        if left < 0 < right:

            def states(k):
                return -pivot.value + np.array(k)

        elif left == 0:

            def states(k):
                return np.array(k)

        elif right == 0:

            def states(k):
                return -pivot.value + np.array(k)

    elif is_levy_copula:
        if method == SamplingMethod.BINARYSEARCHTREEADAPTED:
            return BinarySearchTreeAdapted(model=model, grid=grid)

        raise ValueError(
            "create_sampling_method not yet implemented for levy copula, use SamplingMethod.INVERSION"
        )
    else:
        raise ValueError("Unexpected error when creating the sampling method")

    methods = {
        SamplingMethod.ALIAS: AliasMethod,
        SamplingMethod.TABLE: TableMethod,
        SamplingMethod.BINARYSEARCHTREE: BinarySearchTree,
        SamplingMethod.HUFFMANNTREE: HuffmanTree,
    }

    return methods[method](jump_vector, states)


def create_sampling_inversion_method(grid, model, intensity_of_jumps, is_levy_copula):
    if is_levy_copula:
        if model.dimension() == 2:
            pairing = PairingToZd(pairing=Szudzik(), dimension=model.dimension())
        else:
            # don't use Szudzik which is not a general n-dimensional pairing function
            pairing = PairingToZd(
                pairing=RosenbergStrong(), dimension=model.dimension()
            )
    else:
        left_grid_size = grid.origin_coordinate.value
        right_grid_size = len(grid.axes[0]) - left_grid_size - 1
        pairing = PairingToZ1d((-left_grid_size, right_grid_size), omit_zero=True)

    def probability_to_jump_to_state(state_increment):
        state = grid.origin_coordinate + state_increment
        value = grid[state]
        mid_point_left = grid.middle(grid.left_point(state), value)
        mid_point_right = grid.middle(value, grid.right_point(state))
        state_mass = model.mass(mid_point_left, mid_point_right)
        state_mass = max(state_mass, 0)  # avoid side effects for very small state_mass
        prob = state_mass / intensity_of_jumps
        return prob

    boundary = Boundary()
    domain = Domain(boundary=boundary, grid=grid, pairing=pairing)
    state_manager = StatesManager(pairing=pairing, domain=domain, grid=grid)

    return InversionMethod(
        probability_to_jump_to_state=probability_to_jump_to_state,
        state_manager=state_manager,
    )
