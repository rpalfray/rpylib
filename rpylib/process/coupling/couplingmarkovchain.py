"""Description of a coupling process for a Continuous-Time Markov Chain

The Multi-level Monte-Carlo relies on a process defining the coupling between a 'coarse' process and a 'fine' process
"""

import copy
from types import MethodType

import numpy as np

from ...distribution.sampling import SamplingMethod
from ...distribution.univariate.uniform import Uniform
from ...grid.spatial import CTMCGrid
from ...model.levymodel.levymodel import LevyModel
from ...montecarlo.path import StochasticJumpPath, MCPath
from ...montecarlo.statistic.statistic import PT
from ...product.payoff import PayoffDates
from ...product.product import Product
from ..markovchain.markovchain import MarkovChainProcess
from .helper import create_build_finer_grid_fun


class CouplingMarkovChain:
    """Coupling of a Markov Chain process"""
    def __init__(self, model: LevyModel, method: SamplingMethod, grid: CTMCGrid):
        """
        :param model: Lévy model
        :param grid: grid states
        :param method: sampling method for the discrete distribution (i.e. the distribution over the spatial states)
        """
        self.model = model
        self.grid = grid
        self.method = method
        self.fine_process = MarkovChainProcess(model=model, method=method, grid=grid)
        self._path_coupling_simulation: CouplingSimulation = None

        self.level = 0
        self.uniform = Uniform()
        self.equivalent_diffusion_coefficient_fine = copy.copy(self.fine_process.equivalent_diffusion_coefficient)
        self.equivalent_diffusion_coefficient_coarse = 0

    def one_simulation_cost(self, product) -> float:
        """Cost of simulating one Monte-Carlo path

        :param product: product to price
        """
        cost_fine_process = self.fine_process.one_simulation_cost(product=product)
        cost_coupling = 0
        if self.level > 0:
            # at each jump, we need to simulate a uniform for the coupling, and we need to compute the payoff function
            # for the coarse process (that cost is neglected with regard to the former cost)
            cost_coupling = self.fine_process.intensity_of_jumps
        return cost_fine_process + cost_coupling

    def reset_one_simulation_cost(self) -> None:
        self.uniform.reset_sampling_cost()
        self.fine_process.reset_one_simulation_cost()

    def initialisation(self, product: Product, max_step_epsilon: float = None) -> None:
        self.fine_process.initialisation(product=product, max_step_epsilon=max_step_epsilon)
        if max_step_epsilon is not None:
            self._path_coupling_simulation = CouplingSimulationMaximumStep(self, epsilon=max_step_epsilon)
        elif product.payoff.payoff_dates_type == PayoffDates.DETERMINISTIC:
            self._path_coupling_simulation = CouplingSimulationFixedTimes(self)
        else:
            self._path_coupling_simulation = CouplingSimulationWithJumpTimes(self)

    def pre_computation(self, mc_paths: int,  product: Product) -> None:
        self.fine_process.pre_computation(mc_paths, product)
        self._path_coupling_simulation.pre_computation(mc_paths, product)

    def simulate_one_path(self) -> object:
        return self.fine_process.simulate_one_path()

    def simulate_one_path_with_coupling(self):
        return self._path_coupling_simulation.simulate_one_path_with_coupling()

    def next_level(self, mc_paths: int, path_managers: [MCPath], product: Product, max_step_epsilon: float = None):
        """Update the coupling for the new spatial step h/2

        :param mc_paths: number of Monte-Carlo paths (this is passed to the pre_computation method which initialised
                         the random variables and the cost)
        :param path_managers: list of Monte-Carlo path managers for each MLMC level
        :param product: financial product
        :param max_step_epsilon: maximum step size epsilon for the `SimulationMaximumStep` case
        """
        if path_managers is not None:
            freeze_spots = copy.deepcopy(self.fine_process.deterministic_path(np.zeros(shape=1)))
            freeze_process_drift = copy.deepcopy(self.fine_process.deterministic_path(np.ones(shape=1))) - freeze_spots

            def coarse_deterministic_path(times: np.array) -> np.array:
                return freeze_spots + freeze_process_drift*times

        self.level += 1
        self.grid.refine()
        self.equivalent_diffusion_coefficient_coarse = copy.copy(self.equivalent_diffusion_coefficient_fine)
        self.fine_process = MarkovChainProcess(self.model, method=self.method, grid=self.grid)

        self.initialisation(product=product, max_step_epsilon=max_step_epsilon)
        self.pre_computation(mc_paths=mc_paths, product=product)
        self.equivalent_diffusion_coefficient_fine = self.fine_process.equivalent_diffusion_coefficient

        if path_managers is not None:
            fine_deterministic_path = self.fine_process.deterministic_path
            path_manager_level_l = copy.deepcopy(path_managers[-1])
            path_manager_level_l.update(self.fine_process.process_representation)

            def coupling_deterministic_path(times_input):
                return np.array([fine_deterministic_path(times_input), coarse_deterministic_path(times_input)])
            path_manager_level_l.deterministic_path = coupling_deterministic_path
            path_managers.append(path_manager_level_l)


class CouplingSimulation:
    """Method for the simulation of the coupling process"""
    def __init__(self, coupling_process: CouplingMarkovChain):
        self.coupling_process = coupling_process

    def pre_computation(self, mc_paths: int,  product: Product) -> None:
        raise NotImplementedError

    def simulate_diffusion_with_coupling(self, sqrt_dts):
        w = self.coupling_process.fine_process._path_simulation._brownian_increments.popleft()
        stddev_fine = sqrt_dts*self.coupling_process.equivalent_diffusion_coefficient_fine
        stddev_coarse = sqrt_dts*self.coupling_process.equivalent_diffusion_coefficient_coarse
        diffs_fine = stddev_fine*w
        diffs_coarse = stddev_coarse*w

        return np.cumsum(diffs_fine), np.cumsum(diffs_coarse)

    def simulate_one_path_with_coupling(self):
        raise NotImplementedError

    @staticmethod
    def probability_to_right_jump(grid: CTMCGrid, mass, increment) -> float:
        """Compute the probability to jump to the 'right' state. 'right' means the upper-right corner in the
        multidimensional case.

        :param grid: states grid
        :param mass: mass function
        :param increment: fine process increment
        """
        position = grid.origin_coordinate + increment
        state_value = grid[position]
        mid_point_left = grid.middle(grid.left_point(position), state_value)
        mid_point_right = grid.middle(state_value, grid.right_point(position))
        val_right = mass(state_value, mid_point_right)
        val_left = mass(mid_point_left, state_value)
        probability = val_right/(val_left + val_right)

        return probability

    def coupling_state(self, increment):
        """Coupling implementation. If the increment of the fine process is in the coarse grid, then the coarse
        process has the same increment, otherwise the coupling defines a new increment for the coarse process.

        :param increment: increment of the `fine` process
        :return: the increment for the `coarse` process
        """
        grid = self.coupling_process.grid
        position = grid.origin_coordinate + increment
        if not increment % 2:
            value = grid[position]
        else:
            mass = self.coupling_process.fine_process.model.mass
            if self.coupling_process.uniform.sample() < self.probability_to_right_jump(grid, mass, increment):
                value = grid.right_point(position)
            else:
                value = grid.left_point(position)
        return value

    def coupling_states_for_a_slice(self, slice_fine_states):
        """Apply the coupling for an array of states of the fine process"""
        if slice_fine_states:
            slice_coupling_values = np.empty(shape=len(slice_fine_states), dtype=float)
            current_value = self.coupling_process.grid.origin
            for k, deltaFineState in enumerate(slice_fine_states):
                current_value += self.coupling_state(deltaFineState)
                slice_coupling_values[k] = current_value
        else:
            slice_coupling_values = np.zeros_like(slice_fine_states, dtype=float)

        return slice_coupling_values


class CouplingSimulationFixedTimes(CouplingSimulation):
    """Simulation of the coupling at (pre-defined) fixed times"""

    def __init__(self, coupling_process: CouplingMarkovChain):
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
        fines_states_all_values = fine_mc.values

        fines_states_values = np.zeros(shape=len(fine_states_increments))
        coarse_states_values = np.zeros_like(fines_states_values)

        for k, (slice_fine_states, slice_fine_values) in enumerate(zip(fine_states_increments, fines_states_all_values)):
            if slice_fine_states:
                slice_coarse_values = self.coupling_states_for_a_slice(slice_fine_states)
                fines_states_values[k] = slice_fine_values[-1]
                coarse_states_values[k] = slice_coarse_values[-1]

        return fines_states_values, coarse_states_values

    def simulate_one_path_with_coupling(self):
        # simulate the jump part first
        jumps_h, jumps_2h = self.simulate_jumps_with_coupling()

        # simulate the diffusion part
        diff_h, diff_2h = self.simulate_diffusion_with_coupling(self._sqrt_dts)

        diff = np.zeros(shape=(2, 1+diff_h.shape[0]))
        diff[PT.FP, 1:] = diff_h
        diff[PT.CP, 1:] = diff_2h

        # jumps already have the 0 at t=0
        jumps = np.zeros(shape=(2, 1+jumps_h.shape[0]))
        jumps[PT.FP, 1:] = jumps_h
        jumps[PT.CP, 1:] = jumps_2h

        return StochasticJumpPath(self._times, diff, jumps)


class CouplingSimulationWithJumpTimes(CouplingSimulation):
    """Simulation of the coupling at stochastic jump times"""

    def __init__(self, coupling_process: CouplingMarkovChain):
        super().__init__(coupling_process=coupling_process)
        self._maturity = None

    def pre_computation(self, mc_paths: int,  product: Product) -> None:
        self._maturity = product.maturity

    def simulate_diffusion_with_coupling(self, sqrt_dts):
        w = np.random.normal(size=sqrt_dts.size)
        stddev_fine = sqrt_dts*self.coupling_process.equivalent_diffusion_coefficient_fine
        stddev_coarse = sqrt_dts*self.coupling_process.equivalent_diffusion_coefficient_coarse
        diffs_fine = stddev_fine*w
        diffs_coarse = stddev_coarse*w

        return np.cumsum(diffs_fine), np.cumsum(diffs_coarse)

    def simulate_jumps_with_coupling(self):
        # simulate jump times and jump values of the finer process
        fine_mc = self.coupling_process.fine_process._path_simulation.simulate_markov_chain()
        fine_states_increments = fine_mc.states_increments
        fine_states_all_values = fine_mc.values
        jump_times = fine_mc.times

        coarse_states_all_values = np.empty_like(fine_states_all_values)

        for k, (slice_fine_states, slice_fine_values) in enumerate(zip(fine_states_increments, fine_states_all_values)):
            if slice_fine_states:
                slice_coarse_values = self.coupling_states_for_a_slice(slice_fine_states)
                coarse_states_all_values[k] = slice_coarse_values

        fine_values = np.concatenate(fine_states_all_values).ravel().astype(float)
        coarse_values = np.concatenate(coarse_states_all_values).ravel().astype(float)

        return jump_times, fine_values, coarse_values

    def simulate_one_path_with_coupling(self):
        # simulate the jump part first
        jump_times, jumps_h, jumps_2h = self.simulate_jumps_with_coupling()

        jump_times = np.concatenate(([0.0], + jump_times, [self._maturity]))
        final_fine_jump = 0 if jumps_h.size == 0 else jumps_h[-1]
        final_coarse_jump = 0 if jumps_2h.size == 0 else jumps_2h[-1]
        jumps_h = np.concatenate((+ jumps_h, [final_fine_jump]))
        jumps_2h = np.concatenate((+ jumps_2h, [final_coarse_jump]))

        # simulate the diffusion part
        diff_h, diff_2h = self.simulate_diffusion_with_coupling(np.sqrt(np.diff(jump_times)))

        diff = np.zeros(shape=(2, 1+diff_h.shape[0]))
        diff[0, 1:] = diff_h
        diff[1, 1:] = diff_2h

        # logJumps already have the 0 at t=0
        jumps = np.zeros(shape=(2, 1+jumps_h.shape[0]))
        jumps[0, 1:] = jumps_h
        jumps[1, 1:] = jumps_2h

        return StochasticJumpPath(jump_times, diff, jumps)


class CouplingSimulationMaximumStep(CouplingSimulationWithJumpTimes):
    """Simulation of the coupling at the stochastic jump times but extra time increments are added so that the times
    grid has a maximum time step of size epsilon
    """
    def __init__(self, coupling_process, epsilon: float):
        super().__init__(coupling_process=coupling_process)
        self.epsilon = epsilon

    def pre_computation(self, mc_paths: int,  product: Product) -> None:
        super().pre_computation(mc_paths=mc_paths, product=product)
        build_finer_grid = create_build_finer_grid_fun(epsilon=self.epsilon, maturity=product.maturity)
        self.build_finer_grid = MethodType(build_finer_grid, self)

    def simulate_jumps_with_coupling(self):
        jump_times, fine_all_values, coarse_all_values = super().simulate_jumps_with_coupling()

        if jump_times.size == 0:
            return jump_times, fine_all_values, coarse_all_values
        else:
            return self.build_finer_grid(jump_times, fine_all_values, coarse_all_values)
