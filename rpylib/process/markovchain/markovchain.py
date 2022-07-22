"""One-dimensional CTMC implementation"""


import copy

import numpy as np

from ...distribution.sampling import SamplingMethod
from ...distribution.samplingfactory import create_sampling_method, compute_intensity_of_jumps
from ...grid.spatial import CTMCGrid
from ...model.levycopulamodel import LevyCopulaModel
from ...model.levymodel.levymodel import LevyModel, LevyRepresentation
from ...process.levyprocess import LevyProcess, SimulationFixedTimes, SimulationWithJumpTimes, SimulationMaximumStep, \
    simulate_diffusion_with_brownian_increments
from ...product.payoff import PayoffDates
from ...product.product import Product


def vol_adjustment(model: LevyModel, h: float):
    """For processes with infinite variation, the small jumps of size less than h are approximated by a Brownian motion.
    This function computes the corresponding coefficient.

    :param model: Lévy model
    :param h: spatial step h
    """
    value = 0

    if not model.levy_triplet.nu.jump_of_finite_variation():
        a = max(-h/2, -1)
        b = min(h/2, 1)
        c_h = model.levy_triplet.nu.integrate_against_xx(a, b)
        value = np.sqrt(c_h)
    return value


def compute_mu_h(levy_measure, grid: CTMCGrid, axis: np.array, origin: int) -> float:
    """Computation of the drift mu_h of the CTMC

    :param levy_measure: Lévy measure of the process
    :param grid: states grid
    :param axis: axis
    :param origin: origin coordinate
    :return: the drift mu_h
    """
    integral = levy_measure.integrate
    middle = grid.middle
    right_point = grid.right_point

    mu_h = 0
    mid_point_left = middle(grid.left_point(0), axis[0])
    for position, xi in enumerate(axis):
        if position != origin:
            mid_point_right = middle(xi, right_point(position))
            mu_h += np.array(xi)*integral(mid_point_left, mid_point_right)
            mid_point_left = mid_point_right
        else:
            mid_point_left = middle(grid.left_point(origin+1), axis[origin+1])

    return mu_h


class MarkovChain:
    """Markov Chain object storing the simulation times, its values and state increments."""
    __slots__ = ('times', 'values', 'states_increments')

    def __init__(self, times: np.array, values: list[np.array], states_increments: list[np.array]):
        """
        :param times: simulation times
        :param values: Markov chain values
        :param states_increments: Markov chain increments
        """
        self.times = times
        self.values = values
        self.states_increments = states_increments


class MarkovChainProcess(LevyProcess):
    """Description of a Markov-Chain process approximating a Lévy model over spatial states axes

        :param model: Lévy model
        :param grid: states grid
        :param method: sampling method to simulate the discrete states
    """
    def __init__(self, model: LevyModel, method: SamplingMethod, grid: CTMCGrid):
        model_tilde = copy.deepcopy(model)
        model_tilde.truncate_levy_measure(truncations=grid.truncations[0])
        model_tilde.levy_triplet.set_representation(LevyRepresentation.TILDE)
        super().__init__(model_tilde)

        self.grid = grid
        is_levy_copula = isinstance(model_tilde, LevyCopulaModel)

        self.intensity_of_jumps = compute_intensity_of_jumps(model=model_tilde, grid=grid)
        self.sampling = create_sampling_method(model=model_tilde, levy_measure=model_tilde.levy_triplet.nu,
                                               method=method, grid=grid, is_levy_copula=is_levy_copula,
                                               intensity_of_jumps=self.intensity_of_jumps)
        vol_adj = vol_adjustment(model_tilde, grid.h)
        self.equivalent_diffusion_coefficient = np.sqrt(model_tilde.diffusion_coefficient()**2 + vol_adj**2)
        self._process_drift = None
        self._path_simulation: MCSimulation = MCSimulation()

    def process_drift(self) -> np.array:
        return self._process_drift

    def intensity(self) -> float:
        return self.intensity_of_jumps

    def one_simulation_cost(self, product) -> float:
        cost_markov_chain = np.log(self.grid.number_of_points())
        cost_simulation = self.intensity_of_jumps
        return cost_simulation*(self.model.dimension() + cost_markov_chain)

    def reset_one_simulation_cost(self) -> None:
        pass

    def initialisation(self, product: Product, max_step_epsilon: float = None) -> None:
        if max_step_epsilon is not None:
            self._path_simulation = MCSimulationMaximumStep(self, epsilon=max_step_epsilon)
        elif product.payoff.payoff_dates_type == PayoffDates.DETERMINISTIC:
            self._path_simulation = MCSimulationFixedTimes(self)
        else:
            self._path_simulation = MCSimulationWithJumpTimes(self)

        levy_measure = self.model.levy_triplet.nu
        mu_h = compute_mu_h(levy_measure=self.model.levy_triplet.nu, grid=self.grid, axis=self.grid.axes[0],
                            origin=self.grid.origin_coordinate.value)
        v = 0.0 if self.model.jump_of_finite_variation() else 1.0
        mu_tilde = levy_measure.integrate_against_x(-np.inf, -v) + levy_measure.integrate_against_x(v, np.inf)
        self._process_drift = self.model.drift() + self.model.levy_triplet.a + mu_tilde - mu_h


class MCSimulation:
    """Method to simulation the Markov Chain via Monte-Carlo"""

    def simulate_markov_chain(self) -> MarkovChain:
        raise NotImplementedError

    def simulate_jumps(self):
        raise NotImplementedError

    @staticmethod
    def helper_simulate_markov_chain(grid, sampling, all_nb_of_jumps) -> tuple[list[np.array], list[np.array]]:
        """simulate one path of the non-deterministic part of the underlying

        :param grid: states grid
        :param sampling: sampling method
        :param all_nb_of_jumps: number of jumps for each time interval
        """
        values = [np.empty(shape=nb_of_jumps) for nb_of_jumps in all_nb_of_jumps]
        all_states_increments = [np.empty(shape=nb_of_jumps) for nb_of_jumps in all_nb_of_jumps]
        pivot_position = grid.origin_coordinate
        for k, nb_of_jumps in enumerate(all_nb_of_jumps):
            states_increments = sampling(size=nb_of_jumps)
            values[k] = np.cumsum([grid[pivot_position + increment] for increment in states_increments])
            all_states_increments[k] = states_increments

        return values, all_states_increments


class MCSimulationFixedTimes(MCSimulation, SimulationFixedTimes):
    """Simulation of the Markov Chain at (pre-defined) fixed times"""

    def __init__(self, process: MarkovChainProcess):
        SimulationFixedTimes.__init__(self, process)
        self._grid = process.grid
        self._sampling = process.sampling.sample

    def simulate_markov_chain(self) -> MarkovChain:
        """simulate one path of the non-deterministic part of the underlying
        """
        all_nb_of_jumps = self._poisson_rv.popleft()
        values, all_states_increments = self.helper_simulate_markov_chain(self._grid, self._sampling, all_nb_of_jumps)
        return MarkovChain(self._times[1:], values, all_states_increments)

    @staticmethod
    def project(values):
        definitive_values = np.zeros(shape=len(values), dtype=float)
        for k, sliceStates in enumerate(values):
            if sliceStates.shape[0]:
                definitive_values[k] = sliceStates[-1]
        return definitive_values

    def simulate_jumps(self):
        mc = self.simulate_markov_chain()
        return self.project(mc.values)

    def simulate_diffusion(self, sqrt_dts):
        stddev = sqrt_dts*self.process.equivalent_diffusion_coefficient
        return simulate_diffusion_with_brownian_increments(stddev, self._brownian_increments.popleft())


class MCSimulationWithJumpTimes(MCSimulation, SimulationWithJumpTimes):
    """Simulation of the Markov Chain at stochastic jump times"""

    def __init__(self, process: MarkovChainProcess):
        SimulationWithJumpTimes.__init__(self, process)
        self._grid = process.grid
        self._sampling = process.sampling.sample

    def simulate_markov_chain(self) -> MarkovChain:
        """ simulate one path of the non-deterministic part of the underlying
        """
        jump_times = np.array([])
        all_nb_of_jumps = []
        jump_times_simulation = self.process.jump_times_from_nb_of_jumps
        nb_jump_dt = self.process.nb_jump_dt
        for k, (tp, tm) in enumerate(zip(self._times[1:], self._times)):
            dt = tp - tm
            new_jumps = tm + jump_times_simulation(dt, nb_jump_dt(dt))
            jump_times = np.append(jump_times, new_jumps)
            all_nb_of_jumps.append(new_jumps.size)

        values, all_states_increments = self.helper_simulate_markov_chain(self._grid, self._sampling, all_nb_of_jumps)
        return MarkovChain(jump_times, values, all_states_increments)

    def simulate_jumps(self):
        mc = self.simulate_markov_chain()
        jump_values = np.concatenate(mc.values).ravel().astype(float)
        jump_times = mc.times
        return jump_times, jump_values

    def simulate_diffusion(self, sqrt_dts):
        stddev = sqrt_dts*self.process.equivalent_diffusion_coefficient
        brownian_increments = np.random.normal(size=stddev.size)
        return simulate_diffusion_with_brownian_increments(stddev, brownian_increments)


class MCSimulationMaximumStep(MCSimulationWithJumpTimes, SimulationMaximumStep):
    """Simulation of the Markov Chain at stochastic jump times with maximum time step of size epsilon"""

    def __init__(self, process: MarkovChainProcess, epsilon: float):
        MCSimulationWithJumpTimes.__init__(self, process)
        SimulationMaximumStep.__init__(self, process, epsilon)

    def simulate_jumps(self):
        jump_times, jump_values = super().simulate_jumps()

        if jump_times.size == 0:
            aug_jump_times = jump_times
            aug_jump_values = jump_values
        else:
            aug_jump_times, aug_jump_values = self.build_finer_grid(jump_times, jump_values)

        return aug_jump_times, aug_jump_values
