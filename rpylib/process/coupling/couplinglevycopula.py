"""Description of a coupled process for a Lévy copula

The Multi-level Monte-Carlo uses a coupled process defined as a coarse process and a fine process
"""

import copy
from itertools import product
from types import MethodType

import numpy as np

from ...distribution.sampling import SamplingMethod
from ...distribution.univariate.uniform import Uniform
from ...grid.grid import CoordinateND
from ...grid.spatial import CTMCGrid
from ...model.levycopulamodel import LevyCopulaModel
from ...montecarlo.path import StochasticJumpPath
from ...montecarlo.statistic.statistic import PT
from ...product.payoff import PayoffDates
from ...product.product import Product
from ..markovchain.markovchainlevycopula import MarkovChainLevyCopula
from .helper import create_build_finer_grid_fun


class CouplingProcessLevyCopula:
    """Coupling of a Lévy Copula Markov Chain process"""
    def __init__(self, levy_copula_model: LevyCopulaModel, grid: CTMCGrid, method: SamplingMethod):
        """
        :param levy_copula_model: Lévy copula to be approximated
        :param grid: states grid
        :param method: simulation algorithm method
        """
        self.level = 0
        self.model = levy_copula_model
        self.grid = grid
        self.method = method
        self.fine_process = MarkovChainLevyCopula(levy_copula_model=levy_copula_model, grid=grid, method=method)
        self._path_coupling_simulation: CouplingLevyCopulaSimulation = None

        self._uniform = Uniform()
        self._diffusion_matrix_h = None
        self._diffusion_matrix_2h = None

    def one_simulation_cost(self, product: Product) -> float:
        cost_fine_process = self.fine_process.one_simulation_cost(product=product)
        cost_coupling = 0
        if self.level > 0:
            # at each jump, we need to simulate a uniform for the coupling, and we need to compute the payoff function
            # for the coarse process (the latter cost is neglected with regard to the former)
            cost_coupling = self.fine_process.intensity_of_jumps
        return cost_fine_process + cost_coupling

    def reset_one_simulation_cost(self) -> None:
        self._uniform.reset_sampling_cost()
        self.fine_process.reset_one_simulation_cost()

    def initialisation(self, product: Product, max_step_epsilon: float = None) -> None:
        self.fine_process.initialisation(product=product, max_step_epsilon=max_step_epsilon)
        self._diffusion_matrix_h = self.fine_process._path_simulation.diffusion_matrix
        if max_step_epsilon is not None:
            self._path_coupling_simulation = CouplingLevyCopulaSimulationMaximumStep(self, epsilon=max_step_epsilon)
        elif product.payoff.payoff_dates_type == PayoffDates.DETERMINISTIC:
            self._path_coupling_simulation = CouplingLevyCopulaSimulationFixedTimes(self)
        else:
            self._path_coupling_simulation = CouplingLevyCopulaSimulationWithJumpTimes(self)

    def pre_computation(self, mc_paths: int, product: Product) -> None:
        self.fine_process.pre_computation(mc_paths, product)
        self._path_coupling_simulation.pre_computation(mc_paths, product)

    def simulate_one_path(self) -> object:
        return self.fine_process.simulate_one_path()

    def simulate_one_path_with_coupling(self):
        return self._path_coupling_simulation.simulate_one_path_with_coupling()

    def next_level(self, mc_paths, path_managers, product: Product, max_step_epsilon: float = None):
        if path_managers is not None:
            freeze_spots = copy.deepcopy(self.fine_process.deterministic_path(np.zeros(shape=1)))
            freeze_process_drift = copy.deepcopy(self.fine_process.deterministic_path(np.ones(shape=1))) - freeze_spots

            def coarse_deterministic_path(times: np.array) -> np.array:
                return freeze_spots + freeze_process_drift*times

        self.level += 1
        self.grid.refine()
        self._diffusion_matrix_2h = self.fine_process._path_simulation.diffusion_matrix
        self.fine_process = MarkovChainLevyCopula(levy_copula_model=self.model, grid=self.grid, method=self.method)
        self.initialisation(product=product, max_step_epsilon=max_step_epsilon)
        self.pre_computation(mc_paths=mc_paths, product=product)
        self._diffusion_matrix_h = self.fine_process._path_simulation.diffusion_matrix

        if path_managers is not None:
            fine_deterministic_path = self.fine_process.deterministic_path
            path_manager_levell = copy.deepcopy(path_managers[-1])

            def coupling_deterministic_path(times):
                return np.array([fine_deterministic_path(times), coarse_deterministic_path(times)])
            path_manager_levell.deterministic_path = coupling_deterministic_path
            path_managers.append(path_manager_levell)


class CouplingLevyCopulaSimulation:
    def __init__(self, coupling_process: CouplingProcessLevyCopula):
        self.coupling_process = coupling_process

    def pre_computation(self, mc_paths: int,  product: Product) -> None:
        raise NotImplementedError

    def simulate_diffusion_with_coupling(self, sqrt_dts):
        brownian_increments = self.coupling_process.fine_process._path_simulation._brownian_increments.popleft()
        diff_coefficients_fine = self.coupling_process._diffusion_matrix_h @ brownian_increments
        diff_coefficients_coarse = self.coupling_process._diffusion_matrix_2h @ brownian_increments
        scaled_diff_coefficients_fine = sqrt_dts * diff_coefficients_fine
        scaled_diff_coefficients_coarse = sqrt_dts * diff_coefficients_coarse
        return np.cumsum(scaled_diff_coefficients_fine, axis=1), np.cumsum(scaled_diff_coefficients_coarse, axis=1)

    def simulate_jumps_with_coupling(self):
        raise NotImplementedError

    def simulate_one_path_with_coupling(self):
        raise NotImplementedError

    def __coupling_state(self, increment: tuple[int], axis_coordinates: list[int] = None):
        dim = len(increment)
        if axis_coordinates is None:
            axis_coordinates = list(range(dim))

        grid = self.coupling_process.grid
        if len(axis_coordinates) == 0:
            return grid[grid.origin_coordinate + increment]

        if any(not increment[(j := i)] % 2 for i in axis_coordinates):
            # no projection on this axis
            axis_coordinates.remove(j)
            return self.__coupling_state(increment, axis_coordinates)
        else:
            # projection on the axis defined by the indices in axis_coordinates
            position = grid.origin_coordinate + increment
            mass = self.coupling_process.model.mass
            value = grid[position]
            u = self.coupling_process._uniform.sample()

            projected_position = CoordinateND(position[k] for k in axis_coordinates)
            projected_value = tuple(value[k] for k in axis_coordinates)

            projected_mid_left_value = grid.middle(grid.left_point(projected_position), projected_value)
            projected_mid_right_value = grid.middle(projected_value, grid.right_point(projected_position))
            total_mass = mass(projected_mid_left_value, projected_mid_right_value, axis_coordinates)

            probability = 0
            for p in product([-1, 1], repeat=len(axis_coordinates)):
                p_value = grid[projected_position + p]
                p_middle_value = grid.middle(p_value, projected_value)
                min_max = tuple((min(p1, p2), max(p1, p2)) for p1, p2 in zip(projected_value, p_middle_value))
                p_left_value, p_right_value = zip(*min_max)
                p_mass = mass(p_left_value, p_right_value, axis_coordinates)
                probability += p_mass/total_mass
                if u <= probability:
                    res = tuple(p_value[axis_coordinates.index(k)] if k in axis_coordinates else value[k]
                                for k in range(dim))
                    return np.array(res)

            raise ValueError('couplinglevycopula::__coupling_state -> Numerical error? ')

    def _coupling_states_for_a_slice(self, slice_fine_states: np.array):
        if len(slice_fine_states):
            slice_coupling_values = [None]*len(slice_fine_states)
            current_value = np.array(self.coupling_process.grid.origin)
            for k, deltaFineState in enumerate(slice_fine_states):
                current_value += self.__coupling_state(deltaFineState)
                slice_coupling_values[k] = current_value.copy()
        else:
            slice_coupling_values = [np.array([0.0, 0.0])]*len(slice_fine_states)

        return slice_coupling_values


class CouplingLevyCopulaSimulationFixedTimes(CouplingLevyCopulaSimulation):
    def __init__(self, coupling_process: CouplingProcessLevyCopula):
        super().__init__(coupling_process=coupling_process)
        self._sqrt_dts = None
        self._times = None

    def pre_computation(self, mc_paths: int,  product: Product) -> None:
        times = product.times_grid()
        self._times = times
        self._sqrt_dts = np.sqrt(np.diff(times))

    def simulate_jumps_with_coupling(self):
        # simulate jump times and jump values of the finer process
        fine_mc = self.coupling_process.fine_process._path_simulation.simulate_markov_chain()
        fine_states_increments = fine_mc.states_increments
        fines_states_allvalues = fine_mc.values

        dim = self.coupling_process.fine_process._dimension
        fines_states_values = np.zeros(shape=(dim, len(fine_states_increments)+1))
        coarse_states_values = np.zeros(shape=(dim, len(fine_states_increments)+1))

        for k, (slice_fine_states, slice_fine_values) in enumerate(zip(fine_states_increments, fines_states_allvalues)):
            if slice_fine_states:
                slice_coarse_values = self._coupling_states_for_a_slice(slice_fine_states)
                fines_states_values[:, k+1] = slice_fine_values[-1]
                coarse_states_values[:, k+1] = slice_coarse_values[-1]

        return fines_states_values, coarse_states_values

    def simulate_one_path_with_coupling(self):
        # simulate the jump part first
        jumps_h, jumps_2h = self.simulate_jumps_with_coupling()

        # simulate the diffusion part
        diff_h, diff_2h = self.simulate_diffusion_with_coupling(self._sqrt_dts)

        diff = np.zeros(shape=(2, diff_h.shape[0], 1+diff_h.shape[1]))
        diff[PT.FP, :, 1:] = diff_h
        diff[PT.CP, :, 1:] = diff_2h

        # jumps already have the 0 at t=0
        jumps = np.zeros(shape=(2, jumps_h.shape[0], jumps_h.shape[1]))
        jumps[PT.FP, ...] = jumps_h
        jumps[PT.CP, ...] = jumps_2h

        return StochasticJumpPath(self._times, diff, jumps)


class CouplingLevyCopulaSimulationWithJumpTimes(CouplingLevyCopulaSimulation):
    def __init__(self, coupling_process: CouplingProcessLevyCopula):
        super().__init__(coupling_process=coupling_process)
        self._maturity = None
        self._dimension = self.coupling_process.model.dimension_model()

    def pre_computation(self, mc_paths: int,  product: Product) -> None:
        self._maturity = product.maturity

    def simulate_diffusion_with_coupling(self, sqrt_dts):
        nb = self._dimension*sqrt_dts.size
        brownian_increments = np.random.normal(size=nb).reshape((self._dimension, sqrt_dts.size))
        diff_coeff_fine = self.coupling_process._diffusion_matrix_h @ brownian_increments
        diff_coeff_coarse = self.coupling_process._diffusion_matrix_2h @ brownian_increments
        scaled_diff_coeff_fine = sqrt_dts * diff_coeff_fine
        scaled_diff_coeff_coarse = sqrt_dts * diff_coeff_coarse
        return np.cumsum(scaled_diff_coeff_fine, axis=1), np.cumsum(scaled_diff_coeff_coarse, axis=1)

    def simulate_jumps_with_coupling(self):
        # simulate jump times and jump values of the finer process
        fine_mc = self.coupling_process.fine_process._path_simulation.simulate_markov_chain()
        fine_states_increments = fine_mc.states_increments
        fines_states_allvalues = fine_mc.values
        jump_times = fine_mc.times
        coarse_states_values = np.empty_like(fines_states_allvalues)

        for k, (slice_fine_states, slice_fine_values) in enumerate(zip(fine_states_increments, fines_states_allvalues)):
            if slice_fine_states:
                slice_coarse_values = self._coupling_states_for_a_slice(slice_fine_states)
                coarse_states_values[k] = slice_coarse_values

        fines_states_values = np.concatenate(fines_states_allvalues).T
        coarse_states_values = np.concatenate(coarse_states_values).T

        return jump_times, fines_states_values, coarse_states_values

    def simulate_one_path_with_coupling(self):
        # simulate the jump part first
        jump_times, jumps_h, jumps_2h = self.simulate_jumps_with_coupling()
        jump_times = np.concatenate(([0.0], + jump_times, [self._maturity]))
        final_fine_jump = np.zeros(self._dimension) if jumps_h.size == 0 else np.array([jp[-1] for jp in jumps_h])
        final_coarse_jump = np.zeros(self._dimension) if jumps_2h.size == 0 else np.array([jp[-1] for jp in jumps_2h])

        if jumps_h.size > 0:
            jumps_h = np.concatenate(
                (np.zeros(self._dimension)[:, np.newaxis], jumps_h, final_fine_jump[:, np.newaxis]), axis=1)
        else:
            jumps_h = np.zeros(shape=(self._dimension, 2))
        if jumps_2h.size > 0:
            jumps_2h = np.concatenate(
                (np.zeros(self._dimension)[:, np.newaxis], jumps_2h, final_coarse_jump[:, np.newaxis]), axis=1)
        else:
            jumps_2h = np.zeros(shape=(self._dimension, 2))

        diff_h, diff_2h = self.simulate_diffusion_with_coupling(np.sqrt(np.diff(jump_times)))
        diff = np.zeros(shape=(2, diff_h.shape[0], 1+diff_h.shape[1]))
        diff[PT.FP, :, 1:] = diff_h
        diff[PT.CP, :, 1:] = diff_2h

        # jumps already have the 0 at t=0
        jumps = np.zeros(shape=(2, jumps_h.shape[0], jumps_h.shape[1]))
        jumps[PT.FP, ...] = jumps_h
        jumps[PT.CP, ...] = jumps_2h

        return StochasticJumpPath(jump_times, diff, jumps)


class CouplingLevyCopulaSimulationMaximumStep(CouplingLevyCopulaSimulationWithJumpTimes):
    def __init__(self, coupling_process, epsilon: float):
        super().__init__(coupling_process=coupling_process)
        self.epsilon = epsilon

    def pre_computation(self, mc_paths: int, product: Product) -> None:
        super().pre_computation(mc_paths=mc_paths, product=product)
        build_finer_grid = create_build_finer_grid_fun(epsilon=self.epsilon, maturity=product.maturity)
        self.build_finer_grid = MethodType(build_finer_grid, self)

    def simulate_jumps_with_coupling(self):
        jump_times, fine_all_values, coarse_all_values = super().simulate_jumps_with_coupling()

        if jump_times.size == 0:
            return jump_times, fine_all_values, coarse_all_values
        else:
            return self.build_finer_grid(jump_times, fine_all_values, coarse_all_values)
