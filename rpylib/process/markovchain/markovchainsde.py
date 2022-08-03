"""Lévy-driven SDE process
"""

from functools import partial
from collections.abc import Callable

import numpy as np
import scipy.integrate

from ..markovchain.markovchain import MarkovChainProcess
from ..markovchain.markovchainlevycopula import MarkovChainLevyCopula
from ...distribution.sampling import SamplingMethod
from ...grid.spatial import CTMCGrid
from ...model.levydrivensde.levydrivensde import LevyDrivenSDEModel
from ...model.levydrivensde.levylibormodel import LevyLiborModel
from ...montecarlo.path import StochasticPath, StochasticSDEPath, StochasticJumpPath
from ...process.process import Process, ProcessRepresentation
from ...product.product import Product


class MarkovChainSDE(Process):
    """Markov Chain for a process defined by a Lévy-driven SDE"""

    def __init__(self, model: LevyDrivenSDEModel, method: SamplingMethod, grid: CTMCGrid):
        """
        :param model: Lévy-driven SDE model
        :param method: sampling method of the Markov Chain
        :param grid: states grid
        """
        if model.dimension_model() == 1:
            self.markov_chain = MarkovChainProcess(model=model.driver, method=method, grid=grid)
        else:
            self.markov_chain = MarkovChainLevyCopula(levy_copula_model=model.driver, grid=grid, method=method)

        super().__init__(model=model, process_representation=ProcessRepresentation.Identity)
        self.epsilon = grid.h**model.driver.blumenthal_getoor_index()

    def initialisation(self, product: Product, _: float = None) -> None:
        self.markov_chain.initialisation(product=product, max_step_epsilon=self.epsilon)

    def one_simulation_cost(self, product) -> float:
        # cost = cost of simulating the process + cost of the SDE algorith
        sde_cost = self.markov_chain.intensity_of_jumps
        simulation_cost = self.markov_chain.one_simulation_cost(product)
        return sde_cost + simulation_cost

    def reset_one_simulation_cost(self) -> None:
        self.markov_chain.reset_one_simulation_cost()

    def pre_computation(self, mc_paths: int, product: Product) -> None:
        self.markov_chain.pre_computation(mc_paths=mc_paths, product=product)

    def process_drift(self) -> np.array:
        """Drift of the simulated process"""
        return 0

    def sde_drift(self, t: float, x: np.array):
        return self.model.drift(t, x)

    def deterministic_path(self, times: np.array) -> np.array:
        x = self.model.x0_value()
        return np.broadcast_to(np.array([x]).T, (len(x), times.size))

    def simulate_one_path(self) -> StochasticPath:
        mc_path: StochasticJumpPath = self.markov_chain.simulate_one_path()

        zi = np.array([self.model.x0]).T

        nb_of_jumps = mc_path.jump_times.size
        dimension = self.model.dimension()
        z_jump = np.zeros(shape=(dimension, nb_of_jumps))
        z_diffusion = np.zeros(shape=(dimension, nb_of_jumps))
        z_drift = np.zeros(shape=(dimension, nb_of_jumps))

        a = self.model.a
        mc_drift = self.markov_chain.process_drift()
        sde_drift = self.sde_drift

        for i, (t, dt, dL, dW) in enumerate(zip(mc_path.jump_times[:-1],
                                                np.diff(mc_path.jump_times),
                                                np.diff(mc_path.jump_path).T,
                                                np.diff(mc_path.diffusion_path).T),
                                            start=1):
            sde_drift_val = sde_drift(t, zi)
            a_zi = a(t, zi)

            d_mu = a_zi @ np.atleast_2d(mc_drift)
            d_diffusion = a_zi @ np.atleast_2d(dW).T
            d_jump = a_zi @ np.atleast_2d(dL).T

            drift_dt = (sde_drift_val + d_mu)*dt
            zi += drift_dt + d_jump + d_diffusion
            z_drift[:, i] = drift_dt.flatten()
            z_diffusion[:, i] = d_diffusion.flatten()
            z_jump[:, i] = d_jump.flatten()

        drift = np.cumsum(z_drift, axis=-1)
        diffusion = np.cumsum(z_diffusion, axis=-1)
        jump = np.cumsum(z_jump, axis=-1)

        return StochasticSDEPath(drift, mc_path.jump_times, diffusion, jump)


class MarkovChainLevyLiborModel(MarkovChainSDE):
    """Markov Chain for the Lévy Libor Model and the Forward Market Model

        .. todo:: find a better name
    """

    def __init__(self, model: LevyLiborModel, method: SamplingMethod, grid: CTMCGrid):
        super().__init__(model=model, method=method, grid=grid)
        self.coefficient_sszz = None

    def initialisation(self, product: Product, _: float = None) -> None:
        super().initialisation(product=product)
        self.coefficient_sszz = self._coefficient_sszz()

    def sde_drift(self, t: float, x: np.array):
        x_delta = x.flatten()*self.model.deltas
        omegas = x_delta/(1 + x_delta)
        drift = self.compute_drift_term(t=t, omegas=omegas)
        return -(x.T*drift).T

    def _integral_zz(self):
        driver = self.model.driver
        h = self.markov_chain.grid.h
        left_truncation, right_truncation = self.markov_chain.grid.truncations[0]

        dimension = driver.dimension_model()

        if dimension == 1:
            res2 = driver.levy_triplet.nu.integrate_against_xx(-np.inf, -h/2) \
                   + driver.levy_triplet.nu.integrate_against_xx(h/2, np.inf)
            return np.array([[res2]])

        zz = np.zeros(shape=(dimension, dimension))
        copula = driver.copula
        epsabs = 1e-6

        for i in range(dimension):
            nui = driver.models[i].levy_triplet.nu
            nui_x = nui.x_nu
            zz[i, i] = nui.integrate_against_xx(-np.inf, -h/2) + nui.integrate_against_xx(h/2, np.inf)
            tail_integral_i = partial(driver.marginal_tail_integral, i=i)

            for j in range(i+1, dimension):
                nuj_x = driver.models[j].levy_triplet.nu.x_nu
                tail_integral_j = partial(driver.marginal_tail_integral, i=j)

                def integrand(xx, yy):
                    u = np.array([tail_integral_i(x=xx), tail_integral_j(x=yy)])
                    return nui_x(xx)*nuj_x(yy)*copula.x_first_derivative(u=u)

                res = scipy.integrate.dblquad(func=integrand,
                                              a=left_truncation, b=right_truncation,
                                              gfun=lambda _: left_truncation, hfun=lambda _: right_truncation,
                                              epsabs=epsabs)
                zz[i, j] = zz[j, i] = res[0]

        return zz

    def _coefficient_sszz(self) -> Callable[[float], np.array]:
        nb = self.model._m
        zz = self._integral_zz()

        def helper(t: float) -> np.array:
            res2 = np.zeros(shape=(nb, nb))
            sigma = self.model.a.sigma(t)
            for i in range(nb-1):
                res2[i, i+1:] = sigma[i, :].T @ zz @ sigma[i+1, :]

            return res2

        return helper

    def compute_drift_term(self, t: float, omegas):
        """
        :param t: time
        :param omegas: :math:`\\omega^i = L_t^i*\\delta^i/(1 + L_t^i*\\delta^i)`
        :return: the drift term at order 1
        """
        sszz = self.coefficient_sszz(t)
        order1 = sszz[:, 1:]@omegas[1:]
        return order1
