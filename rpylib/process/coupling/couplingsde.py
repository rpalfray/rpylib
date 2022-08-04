"""Description of a coupled process for a Lévy-driven SDE

"""

import copy

import numpy as np

from .couplinglevycopula import CouplingProcessLevyCopula
from .couplingmarkovchain import CouplingMarkovChain
from ..markovchain.markovchainsde import MarkovChainSDE, MarkovChainLevyLiborModel
from ...distribution.sampling import SamplingMethod
from ...grid.spatial import CTMCGrid
from ...model.levydrivensde.levydrivensde import LevyDrivenSDEModel
from ...model.levydrivensde.levylibormodel import LevyLiborModel
from ...montecarlo.path import StochasticSDEPath
from ...product.product import Product


class CouplingSDE:

    def __init__(self, model: LevyDrivenSDEModel, grid: CTMCGrid, method: SamplingMethod):
        self.level = 0
        self.model = model
        self.method = method
        self.epsilon = grid.h**model.driver.blumenthal_getoor_index()

        # only for level 0 - note that, by design, it must be named self.fine_process
        if isinstance(model, LevyLiborModel):  # FIXME: not elegant
            self.fine_process = MarkovChainLevyLiborModel(model=model, method=method, grid=grid)
        else:
            self.fine_process = MarkovChainSDE(model=model, method=method, grid=grid)

        self._process_representation = self.fine_process.process_representation
        self._spots = self.fine_process.deterministic_path(np.zeros(shape=1))

        if model.dimension_model() == 1:
            self.driver_coupling_process = CouplingMarkovChain(model=model.driver, method=method, grid=grid)
        else:
            self.driver_coupling_process = CouplingProcessLevyCopula(levy_copula_model=model.driver, grid=grid,
                                                                     method=method)

        self.mc_drift_h = None
        self.mc_drift_2h = None

    def one_simulation_cost(self, product) -> float:
        cost_fine_process = self.fine_process.one_simulation_cost(product=product)
        cost_coupling = 0
        if self.level > 0:
            # for each path, the cost is the cost of simulating the process + the cost of the SDE algorithm
            # which is in big O of intensity of jumps and which term is already present in the former one.
            cost_coupling = self.driver_coupling_process.one_simulation_cost(product=product)
        return cost_fine_process + cost_coupling

    def reset_one_simulation_cost(self) -> None:
        self.fine_process.reset_one_simulation_cost()
        self.driver_coupling_process.reset_one_simulation_cost()

    def initialisation(self, product: Product) -> None:
        if self.level == 0:
            self.fine_process.initialisation(product=product)
            self.mc_drift_h = self.fine_process.markov_chain.process_drift()
        else:
            self.driver_coupling_process.initialisation(product=product, max_step_epsilon=self.epsilon)

    def pre_computation(self, mc_paths: int, product: Product) -> None:
        if self.level == 0:
            self.fine_process.pre_computation(mc_paths=mc_paths, product=product)
        else:
            self.driver_coupling_process.pre_computation(mc_paths=mc_paths, product=product)

    def simulate_one_path(self) -> object:
        return self.fine_process.simulate_one_path()

    def simulate_one_path_with_coupling(self):
        mc_path = self.driver_coupling_process.simulate_one_path_with_coupling()
        nb_of_jumps = mc_path.jump_times.size
        dimension = self.model.dimension()

        zi = np.stack((np.array([self.model.x0]).T, np.array([self.model.x0]).T))
        z_drift = np.zeros(shape=(2, dimension, nb_of_jumps))
        z_diffusion = np.zeros(shape=(2, dimension, nb_of_jumps))
        z_jump = np.zeros(shape=(2, dimension, nb_of_jumps))

        a = self.model.a
        sde_drift = self.fine_process.sde_drift
        mc_drift = np.stack((np.atleast_2d(self.mc_drift_h), np.atleast_2d(self.mc_drift_2h)))

        for i, (t, dt, dL, dW) in enumerate(zip(mc_path.jump_times[:-1],
                                                np.diff(mc_path.jump_times),
                                                np.diff(mc_path.jump_path).T,
                                                np.diff(mc_path.diffusion_path).T),
                                            start=1):
            sde_drift_val = np.stack((sde_drift(t, zi[0]), sde_drift(t, zi[1])))
            a_zi = a(t, zi)

            d_mu = a_zi @ mc_drift
            d_diffusion = a_zi @ np.atleast_2d(dW).T[..., np.newaxis]
            d_jump = a_zi @ np.atleast_2d(dL).T[..., np.newaxis]

            drift_dt = (sde_drift_val + d_mu) * dt
            zi += drift_dt + d_jump + d_diffusion

            z_drift[..., i] = drift_dt.reshape((2, dimension))
            z_diffusion[..., i] = d_diffusion.reshape((2, dimension))
            z_jump[..., i] = d_jump.reshape((2, dimension))

        drift = np.cumsum(z_drift, axis=-1)
        diffusion = np.cumsum(z_diffusion, axis=-1)
        jump = np.cumsum(z_jump, axis=-1)

        return StochasticSDEPath(drift, mc_path.jump_times, diffusion, jump)

    def next_level(self, mc_paths, path_managers, product: Product, _: float = None):
        self.level += 1
        self.epsilon = (self.driver_coupling_process.grid.h/2) ** self.model.driver.blumenthal_getoor_index()
        self.mc_drift_2h = copy.deepcopy(self.mc_drift_h)
        self.initialisation(product=product)
        # note that the grid is refined in the next call:
        self.driver_coupling_process.next_level(mc_paths=mc_paths, path_managers=None, product=product,
                                                max_step_epsilon=self.epsilon)
        self.mc_drift_h = self.driver_coupling_process.fine_process.process_drift()

        path_manager_level_l = copy.deepcopy(path_managers[-1])
        path_manager_level_l.update(self._process_representation)

        # the drift is directly dealt with in the simulation function, hence the following function for both the coarse
        # and fine processes:
        def _deterministic_path(_: np.array) -> np.array:
            return self._spots

        def coupling_deterministic_path(times):
            return np.array([_deterministic_path(times), _deterministic_path(times)])

        path_manager_level_l.deterministic_path = coupling_deterministic_path
        path_managers.append(path_manager_level_l)
