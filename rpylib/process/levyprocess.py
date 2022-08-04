"""Definition of a stochastic process for a Lévy model
"""

from collections import deque
from types import MethodType
from typing import Union

import numpy as np

from .process import Process
from ..distribution.univariate.poisson import Poisson
from ..distribution.univariate.uniform import Uniform
from ..grid.time import TimeGrid
from ..model.levycopulamodel import LevyCopulaModel
from ..model.levymodel.levymodel import LevyModel
from ..montecarlo.path import StochasticPath, StochasticJumpPath
from ..product.payoff import PayoffDates
from ..product.product import Product


def simulate_diffusion_with_brownian_increments(scaled_stddev, brownian_increments):
    diffs = scaled_stddev*brownian_increments
    return np.cumsum(diffs)


class LevyProcess(Process):
    """Defines a simulation process for jump models (with a diffusive part and a pure jump part)

    There are 2 ways to simulate the jumps:
        1. simulate the jumping times and then simulate the corresponding jump values
        2. simulate the number of jumps between t and s and then simulate the jump values

    When the payoff has stochastic dates (for example for credit models) or the whole path is needed (barrier option)
    then option 1. is chosen otherwise we just go with option 2. as the simulation is slightly faster (because there is
    no need to simulate the jump times)
    """
    def __init__(self, model: Union[LevyModel, LevyCopulaModel]):
        super().__init__(model=model, process_representation=model.process_representation)
        self.model = model
        self._path_simulation: Simulation = Simulation(self)
        self._uniform = Uniform()

    def intensity(self):
        """Intensity of the jump process"""
        return self.model.intensity()

    def nb_jump_dt(self, dt) -> int:
        """Direct simulation of the numbers of jumps between 0 and dt

        :param dt: time increment
        :return: number of jumps
        """
        poisson = Poisson(dt*self.intensity())
        res = poisson.sample()[0]
        return res

    def jump_times(self, dt: float) -> np.array:
        """Direct simulation of the jump times

        :param dt: time increment
        :return: array of the jump times
        """
        t = 0
        jump_times = []
        rdm, intensity = self._uniform, self.intensity()
        while True:
            t += -np.log(rdm.sample())/intensity
            if t < dt:
                jump_times.append(t)
            else:
                break
        return np.array(jump_times)

    @staticmethod
    def jump_times_from_nb_of_jumps(dt: float, n: int) -> np.array:
        """Simulation of n jump times in [0, dt)

        :param dt: time increment
        :param n: number of jumps
        :return: jump times
        """
        res = dt*np.random.random_sample(size=n)
        return np.sort(res)

    def one_simulation_cost(self, product) -> float:
        """
        :param product: product to price
        :return: the cost of simulating one Monte-Carlo path
        """
        return 0

    def reset_one_simulation_cost(self):
        """Set simulation cost to 0"""
        pass

    def initialisation(self, product: Product, max_step_epsilon: float = None):
        """Initialisation of auxiliary objects"""
        if max_step_epsilon is not None:
            self._path_simulation = SimulationMaximumStep(self, epsilon=max_step_epsilon)
        elif product.payoff.payoff_dates_type == PayoffDates.DETERMINISTIC:
            self._path_simulation = SimulationFixedTimes(self)
        else:
            self._path_simulation = SimulationWithJumpTimes(self)

    def pre_computation(self, mc_paths: int, product: Product):
        """Pre-computation of random variables and other quantities"""
        self._path_simulation.pre_computation(mc_paths, product)

    def simulate_one_path(self) -> StochasticPath:
        """Simulation of one Monte-Carlo path"""
        return self._path_simulation.simulate_one_path()


class Simulation:
    """Simulation method"""

    def __init__(self, process: LevyProcess):
        self.process = process

    def one_simulation_cost(self, product) -> float:
        """Cost of the simulation of onr Monte-Carlo path"""
        return 0

    def reset_one_simulation_cost(self):
        """Set the simulation cost to 0"""
        pass

    def pre_computation(self, mc_paths: int, product: Product):
        """Pre-compute auxiliary objevts"""
        raise NotImplementedError

    def simulate_one_path(self) -> StochasticPath:
        """Simulate one Monte-Carlo path"""
        raise NotImplementedError

    def simulate_jumps(self) -> np.array:
        """Simulation of the jump component"""
        raise NotImplementedError

    def simulate_diffusion(self, sqrt_dts) -> np.array:
        """Simulation of the diffusive component"""
        raise NotImplementedError


class SimulationFixedTimes(Simulation):
    """Simulation of the process at (predefined) fixed times"""

    def __init__(self, process: LevyProcess):
        super().__init__(process)
        self._brownian_increments = deque()
        self._poisson_rv = deque()
        self._times = None
        self._sqrt_dts = None

    def pre_computation(self, mc_paths: int, product: Product):
        times = product.times_grid()
        nb = len(times) - 1
        self._times = times
        self._sqrt_dts = np.sqrt(np.diff(times))

        # the Brownian increments are simulated directly at each payoff dates
        poisson_np_array = np.empty(shape=(mc_paths, nb), dtype=int)
        nb_jump_dt = self.process.nb_jump_dt
        for k, (tp, tm) in enumerate(zip(times[1:], times)):
            dt = tp - tm
            poisson_np_array[:, k] = [nb_jump_dt(dt) for _ in range(mc_paths)]
        self._brownian_increments = deque(np.random.normal(size=(mc_paths, self.process.dimension(), nb)).tolist())
        self._poisson_rv = deque(poisson_np_array.tolist())

    def simulate_one_path(self) -> StochasticPath:
        """Simulation of the path for deterministic payoff dates"""
        # simulate the jump values
        jumps = np.concatenate(([0.0], + self.simulate_jumps()))

        # simulate the diffusion part
        diff = np.concatenate(([0.0], + self.simulate_diffusion(self._sqrt_dts)))

        return StochasticJumpPath(self._times, diff, jumps)

    def simulate_jumps(self):
        all_nb_of_jumps = self._poisson_rv.popleft()
        jump_increment = self.process.model.jump_increment
        increments = [jump_increment(n=nbOfJumps) for nbOfJumps in all_nb_of_jumps]
        jump_values = np.array([np.sum(increment) for increment in increments])
        return jump_values

    def simulate_diffusion(self, sqrt_dts):
        stddev = sqrt_dts*self.process.model.diffusion_coefficient()
        return simulate_diffusion_with_brownian_increments(stddev, self._brownian_increments.popleft())


class SimulationWithJumpTimes(Simulation):
    """Simulation of the process at the stochastic jump times"""

    def __init__(self, process: LevyProcess):
        super().__init__(process)
        self._times: TimeGrid = None
        self._maturity: float = None

    def pre_computation(self, mc_paths: int, product: Product):
        # There is potentially a lot of jumps, and it might not be possible to pre-computed there because
        # of memory limitations, therefore jumps and Brownian increments are simulated on the fly.
        self._times = product.times_grid()
        self._maturity = product.maturity

    def simulate_one_path(self) -> StochasticPath:
        """Simulation of the path for stochastic payoff dates"""
        # simulate the jump values
        jump_times, jumps = self.simulate_jumps()
        # add values for t=0 and t=maturity
        jump_times = np.concatenate(([0.0], + jump_times, [self._maturity]))
        final_jump = 0 if jumps.size == 0 else jumps[-1]
        jumps = np.concatenate(([0.0], + jumps, [final_jump]))

        # simulate the diffusion part
        sqrt_dts = np.sqrt(np.diff(jump_times))
        diff = np.concatenate(([0.0], + self.simulate_diffusion(sqrt_dts)))

        return StochasticJumpPath(jump_times, diff, jumps)

    def simulate_jumps(self):
        jump_times = np.empty(shape=0, dtype=float)
        increments = np.empty(shape=0, dtype=float)
        jump_times_simulation = self.process.jump_times_from_nb_of_jumps
        nb_jump_dt = self.process.nb_jump_dt
        jump_increment = self.process.model.jump_increment
        for k, (tp, tm) in enumerate(zip(self._times[1:], self._times)):
            dt = tp - tm
            new_jumps = tm + jump_times_simulation(dt, nb_jump_dt(dt))
            jump_times = np.append(jump_times, new_jumps)
            increments = np.append(increments, [jump_increment(n=jp.size) for jp in new_jumps])
        jump_values = np.cumsum(increments)

        return jump_times, jump_values

    def simulate_diffusion(self, sqrt_dts):
        stddev = sqrt_dts*self.process.model.diffusion_coefficient()
        brownian_increments = np.random.normal(size=stddev.size)
        return simulate_diffusion_with_brownian_increments(stddev, brownian_increments)


class SimulationMaximumStep(SimulationWithJumpTimes):
    """Simulation of the path at the stochastic times of the path but a step size of size epsilon is added for any
    time increment greater than epsilon (that is the maximum times grid is epsilon)
     """
    def __init__(self, process: LevyProcess, epsilon: float):
        """
        :param process: Lévy process
        :param epsilon: minimum time increment
        """
        super().__init__(process)
        self.epsilon = epsilon

    @staticmethod
    def create_build_finer_grid_fun(epsilon: float, maturity: float):
        def _build_finer_grid_default(self, jump_times, jump_values):
            return jump_times, jump_values

        def _build_finer_grid(self, jump_times, jump_values):
            dts = np.diff(jump_times, prepend=0)
            if not any(dts > epsilon):
                return jump_times, jump_values
            else:
                positions = np.flatnonzero(dts > epsilon)
                aug_dts = dts
                aug_jump_values = jump_values
                while positions.size > 0:
                    aug_dts[positions] -= epsilon
                    aug_dts = np.insert(aug_dts, positions, epsilon)
                    aug_jump_values = np.insert(aug_jump_values, positions,
                                                np.where(positions == 0, 0, aug_jump_values[..., positions - 1]),
                                                axis=-1)
                    positions = np.flatnonzero(aug_dts > epsilon)
                aug_jump_times = np.cumsum(aug_dts)

                return aug_jump_times, aug_jump_values

        return _build_finer_grid_default if epsilon >= maturity else _build_finer_grid

    def pre_computation(self, mc_paths: int, product: Product):
        super().pre_computation(mc_paths=mc_paths, product=product)
        build_finer_grid = self.create_build_finer_grid_fun(epsilon=self.epsilon, maturity=product.maturity)
        self.build_finer_grid = MethodType(build_finer_grid, self)

    def simulate_jumps(self):
        jump_times, jump_values = super().simulate_jumps()

        if jump_times.size == 0:
            return jump_times, jump_values
        else:
            return self.build_finer_grid(jump_times, jump_values)
