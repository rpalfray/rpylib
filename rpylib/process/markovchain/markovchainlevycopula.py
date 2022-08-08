"""
Lévy copula simulated through a Markov Chain process

"""

import copy

import numpy as np
import pathos.multiprocessing as mp
import scipy.integrate
import scipy.linalg

from rpylib.distribution.sampling import SamplingMethod
from rpylib.distribution.samplingfactory import (
    create_sampling_method,
    compute_intensity_of_jumps,
)
from rpylib.grid.spatial import CTMCGrid
from rpylib.model.levycopulamodel import LevyCopulaModel
from rpylib.model.levymodel.levymodel import LevyRepresentation
from rpylib.montecarlo.path import StochasticJumpPath, StochasticPath
from rpylib.numerical.tools import sign
from rpylib.process.levyprocess import LevyProcess
from rpylib.process.levyprocess import (
    SimulationFixedTimes,
    SimulationWithJumpTimes,
    SimulationMaximumStep,
)
from rpylib.process.markovchain.markovchain import MarkovChain, compute_mu_h
from rpylib.product.payoff import PayoffDates
from rpylib.product.product import Product


def vol_adjustment_ij(i: int, j: int, h: float, levy_model: LevyCopulaModel):
    """For processes with infinite variation, the small jumps of size less than h are approximated by a Brownian motion.
    This function computes the corresponding diffusion matrix.

    :param i: i-th coordinate
    :param j: j-th coordinate
    :param levy_model: Lévy copula model
    :param h: spatial step h
        .. seealso:: function :func:`vol_adjustment`
    """

    V = 0.0 if levy_model.jump_of_finite_variation() else 1.0
    a = [max(-V, -h / 2)] * levy_model.dimension()
    b = [min(+V, +h / 2)] * levy_model.dimension()
    Ah = list(zip(a, b))
    mass = levy_model.mass

    def to_integrate_ii(*s):
        if (s_i := s[i]) == 0:
            return 0
        else:
            a[i], b[i] = (s_i, h / 2) if s_i > 0 else (-h / 2, s_i)
            val = sign(s_i) * s_i * mass(a=a, b=b)
            return val

    def to_integrate_ij(*s):
        s_i, s_j = s[i], s[j]
        if s_i == 0 or s_j == 0:
            return 0
        else:
            a[i], b[i] = (s_i, h / 2) if s_i > 0 else (-h / 2, s_i)
            a[j], b[j] = (s_j, h / 2) if s_j > 0 else (-h / 2, s_j)
            val = sign(s_i) * sign(s_j) * mass(a=a, b=b)
            return val

    options = {"epsabs": 1e-3}
    if i == j:
        root = scipy.integrate.nquad(func=to_integrate_ii, ranges=Ah, opts=options)
        res = root[0] * 2 / h ** (levy_model.dimension() - 1)
    else:
        root = scipy.integrate.nquad(func=to_integrate_ij, ranges=Ah, opts=options)
        res = root[0] / h ** (levy_model.dimension() - 2)
    return res


class MarkovChainLevyCopula(LevyProcess):
    """Lévy copula simulated via a Markov Chain process"""

    def __init__(
        self, levy_copula_model: LevyCopulaModel, grid: CTMCGrid, method: SamplingMethod
    ):
        """
        :param levy_copula_model: Lévy copula model
        :param grid: states grid
        :param method: sampling method
        """
        model_tilde = copy.deepcopy(levy_copula_model)
        model_tilde.truncate_levy_measure(truncations=grid.truncations)
        for model in model_tilde.models:
            model.levy_triplet.set_representation(LevyRepresentation.TILDE)
        super().__init__(model=model_tilde)
        self.grid = grid
        self.method = method

        self.intensity_of_jumps = compute_intensity_of_jumps(
            model=model_tilde, grid=grid
        )
        self.sampling = create_sampling_method(
            model=model_tilde,
            levy_measure=None,
            method=method,
            grid=grid,
            is_levy_copula=True,
            intensity_of_jumps=self.intensity_of_jumps,
        )

        self._dimension = model_tilde.dimension()
        self._h = grid.h
        self._process_drift = None
        self._path_simulation: MCLevyCopulaSimulation = MCLevyCopulaSimulation(
            process=self
        )

    def process_drift(self) -> np.array:
        return self._process_drift

    def intensity(self) -> float:
        return self.intensity_of_jumps

    def one_simulation_cost(self, product) -> float:
        cost_markov_chain = np.log(self.grid.number_of_points())
        cost_simulation = self.intensity_of_jumps
        return cost_simulation * (self.model.dimension() + cost_markov_chain)

    def reset_one_simulation_cost(self) -> None:
        super().reset_one_simulation_cost()
        self.sampling.reset_sampling_cost()

    def initialisation(self, product: Product, max_step_epsilon: float = None) -> None:
        if max_step_epsilon is not None:
            self._path_simulation = MCLevyCopulaSimulationMaximumStep(
                self, max_step_epsilon
            )
        elif product.payoff.payoff_dates_type == PayoffDates.DETERMINISTIC:
            self._path_simulation = MCLevyCopulaSimulationFixedTimes(self)
        else:
            self._path_simulation = MCLevyCopulaSimulationWithJumpTimes(self)

        models = self.model.models
        mu_h = np.array(
            [
                [
                    compute_mu_h(
                        levy_measure=model.levy_triplet.nu,
                        grid=self.grid,
                        axis=self.grid.axes[k],
                        origin=self.grid.origin_coordinate.value[k],
                    )
                    for k, model in enumerate(models)
                ]
            ]
        ).T
        V = 0.0 if self.model.jump_of_finite_variation() else 1.0
        mu_tilde = np.array(
            [
                [
                    model.levy_triplet.nu.integrate_against_x(-np.inf, -V)
                    + model.levy_triplet.nu.integrate_against_x(V, np.inf)
                    for model in models
                ]
            ]
        ).T
        a_drift = np.array([[model.levy_triplet.a for model in models]]).T
        self._process_drift = self.model.drift() + a_drift + mu_tilde - mu_h


class MCLevyCopulaSimulation:
    """Simulation method of Markov Chain process for Lévy copulas"""

    def __init__(self, process: MarkovChainLevyCopula):
        self.process = process
        models = process.model.models
        model = process.model
        dimension = model.dimension()
        h = process.grid.h
        model_variance = np.diag([m.diffusion_coefficient() ** 2 for m in models])
        adj_matrix = np.zeros_like(model_variance)

        if not model.jump_of_finite_variation():
            nb_of_processes = ((dimension + 1) * dimension) // 2
            with mp.Pool(processes=nb_of_processes) as pool:
                res = [
                    pool.apply_async(vol_adjustment_ij, args=(i, j, h, model))
                    for i in range(dimension)
                    for j in range(i, dimension)
                ]
                outputs = iter([p.get() for p in res])

            for i in range(dimension):
                adj_matrix[i, i] = next(outputs)
                for j in range(i + 1, dimension):
                    adj_matrix[i, j] = adj_matrix[j, i] = next(outputs)

        variance_matrix = np.dot(adj_matrix, adj_matrix.T) + model_variance
        diffusion_matrix = scipy.linalg.sqrtm(variance_matrix)
        self.diffusion_matrix = diffusion_matrix

    def simulate_markov_chain(self) -> MarkovChain:
        raise NotImplementedError

    def simulate_jumps(self):
        raise NotImplementedError

    def helper_simulate_levy_copula_markov_chain(
        self, all_nb_of_jumps
    ) -> tuple[list[np.array], list[np.array]]:
        """simulate one path of the non-deterministic part of the underlying"""
        all_values = [np.empty(shape=nb_of_jumps) for nb_of_jumps in all_nb_of_jumps]
        all_states_increments = [
            np.empty(shape=nb_of_jumps) for nb_of_jumps in all_nb_of_jumps
        ]
        grid = self.process.grid
        sample = self.process.sampling.sample
        pivot_position = grid.origin_coordinate
        for k, nb_of_jumps in enumerate(all_nb_of_jumps):
            states_increments = sample(size=nb_of_jumps)
            xy = (
                grid[pivot_position + state_increment]
                for state_increment in states_increments
            )
            all_values[k] = np.cumsum(np.array(list(xy)), axis=0)
            all_states_increments[k] = states_increments

        return all_values, all_states_increments

    def helper_simulate_diffusion_part(self, sqrt_time_increments, brownian_increments):
        diff_coefficient = self.diffusion_matrix @ brownian_increments
        scaled_diff_coefficient = sqrt_time_increments * diff_coefficient
        return np.cumsum(scaled_diff_coefficient, axis=1)


class MCLevyCopulaSimulationFixedTimes(MCLevyCopulaSimulation, SimulationFixedTimes):
    """Simulation of a Markov Chain for Lévy copula at (pre-defined) fixed times"""

    def __init__(self, process: MarkovChainLevyCopula):
        MCLevyCopulaSimulation.__init__(self, process)
        SimulationFixedTimes.__init__(self, process)
        self._dimension = self.process.model.dimension()

    def simulate_one_path(self) -> StochasticPath:
        # simulate the jump values
        simulated_jumps = self.simulate_jumps()
        jumps = np.hstack(
            (np.zeros(self._dimension)[:, np.newaxis], simulated_jumps[:, np.newaxis])
        )

        # simulate the diffusion part
        simulated_diffusion = self.simulate_diffusion_part()
        diffusion = np.hstack(
            (np.zeros(self._dimension)[:, np.newaxis], simulated_diffusion)
        )

        return StochasticJumpPath(self._times, diffusion, jumps)

    def simulate_markov_chain(self) -> MarkovChain:
        """simulate one path of the non-deterministic part of the underlying"""
        all_nb_of_jumps = self._poisson_rv.popleft()
        values, all_states_increments = self.helper_simulate_levy_copula_markov_chain(
            all_nb_of_jumps
        )
        return MarkovChain(self._times[1:], values, all_states_increments)

    @staticmethod
    def project(values, dim):
        zero = (0.0,) * dim
        definitive_values = (
            sliceStates[-1] if sliceStates.size else zero for sliceStates in values
        )
        return np.array(*definitive_values)

    def simulate_jumps(self):
        mc = self.simulate_markov_chain()
        return self.project(mc.values, self._dimension)

    def simulate_diffusion_part(self):
        brownian_increments = self._brownian_increments.popleft()
        return self.helper_simulate_diffusion_part(self._sqrt_dts, brownian_increments)


class MCLevyCopulaSimulationWithJumpTimes(
    MCLevyCopulaSimulation, SimulationWithJumpTimes
):
    """Simulation of a Markov Chain for Lévy copula at the stochastic jump times"""

    def __init__(self, process: MarkovChainLevyCopula):
        MCLevyCopulaSimulation.__init__(self, process)
        SimulationWithJumpTimes.__init__(self, process)
        self._dimension = self.process.model.dimension()

    def simulate_one_path(self) -> StochasticPath:
        # simulate the jump values
        jump_times, jump_values = self.simulate_jumps()
        # add values for t=0 and t=maturity
        jump_times = np.concatenate(([0.0], +jump_times, [self._maturity]))
        final_jumps = (
            np.zeros(self._dimension)
            if jump_values.size == 0
            else np.array([jp[-1] for jp in jump_values])
        )

        if jump_values.size > 0:
            jumps = np.concatenate(
                (
                    np.zeros(self._dimension)[:, np.newaxis],
                    jump_values,
                    final_jumps[:, np.newaxis],
                ),
                axis=1,
            )
        else:
            jumps = np.zeros(shape=(self._dimension, 2))

        simulated_diffusion = self.simulate_diffusion_part(np.sqrt(np.diff(jump_times)))
        diffusion = np.concatenate(
            (np.zeros(self._dimension)[:, np.newaxis], simulated_diffusion), axis=1
        )

        return StochasticJumpPath(jump_times, diffusion, jumps)

    def simulate_markov_chain(self) -> MarkovChain:
        """simulate one path of the non-deterministic part of the underlying"""
        jump_times = np.array([])
        all_nb_of_jumps = []
        jump_times_simulation = self.process.jump_times_from_nb_of_jumps
        nb_jump_dt = self.process.nb_jump_dt
        for k, (tp, tm) in enumerate(zip(self._times[1:], self._times)):
            dt = tp - tm
            new_jumps = jump_times_simulation(dt, nb_jump_dt(dt)) + tm
            jump_times = np.append(jump_times, new_jumps)
            all_nb_of_jumps.append(new_jumps.size)

        values, all_states_increments = self.helper_simulate_levy_copula_markov_chain(
            all_nb_of_jumps
        )
        return MarkovChain(jump_times, values, all_states_increments)

    def simulate_jumps(self):
        mc = self.simulate_markov_chain()
        jump_values = np.concatenate(mc.values, axis=-1).T
        jump_times = mc.times
        return jump_times, jump_values

    def simulate_diffusion_part(self, sqrt_time_increments):
        nb = self._dimension * sqrt_time_increments.size
        brownian_increments = np.random.normal(size=nb).reshape(
            (self._dimension, sqrt_time_increments.size)
        )
        return self.helper_simulate_diffusion_part(
            sqrt_time_increments, brownian_increments
        )


class MCLevyCopulaSimulationMaximumStep(
    MCLevyCopulaSimulationWithJumpTimes, SimulationMaximumStep
):
    """Simulation of a Markov Chain for Lévy copula at the stochastic jump times with maximum time increment of
    size epsilon"""

    def __init__(self, process: MarkovChainLevyCopula, epsilon: float):
        MCLevyCopulaSimulationWithJumpTimes.__init__(self, process)
        SimulationMaximumStep.__init__(self, process, epsilon)

    def simulate_jumps(self):
        jump_times, jump_values = MCLevyCopulaSimulationWithJumpTimes.simulate_jumps(
            self
        )

        if jump_times.size == 0:
            return jump_times, jump_values
        else:
            return self.build_finer_grid(jump_times, jump_values)
