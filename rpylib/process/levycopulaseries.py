"""Lévy copula simulated via a series representation as described in 'Lévy copulas: review of recent results',
by Peter Tankov

This implementation only supports 2d Levy processes with fine variation.
"""

import numpy as np

from .process import Process
from ..distribution.univariate.poisson import Poisson
from ..distribution.univariate.uniform import Uniform
from ..model.levycopulamodel import LevyCopulaModel
from ..montecarlo.path import StochasticJumpPath
from ..product.product import Product


class LevyCopula2dSeriesRepresentation(Process):
    """Simulation of a Lévy copula via a series representation

        .. note::: the Lévy process is a 2d dimensional Lévy copula with finite variation
    """
    def __init__(self, levy_copula_model: LevyCopulaModel, tau: float):
        """
        :param levy_copula_model: Lévy copula model
        :param tau: cut-off of the series representation
        """
        super().__init__(model=levy_copula_model, process_representation=levy_copula_model.process_representation)

        if levy_copula_model.dimension_model() != 2:
            raise ValueError('Series representation process not implemented for dimension > 2')

        if not levy_copula_model.jump_of_finite_variation():
            raise ValueError('The series representation approximation is only implemented for levy processes '
                             'with finite variation')

        self.levy_copula_model = levy_copula_model
        self.model1 = levy_copula_model.models[0]
        self.model2 = levy_copula_model.models[1]
        self.tau = tau

        self.times = np.zeros(shape=0)
        self._process_drift = None

        self.stddev_grid_1: np.array = None
        self.stddev_grid_2: np.array = None

    def process_drift(self) -> np.array:
        return self._process_drift

    def bound_supremum(self):
        nu1, nu2 = self.model1.levy_triplet.nu, self.model2.levy_triplet.nu

        ivt = self.levy_copula_model.inverse_tail_integral

        a1, b1 = ivt(i=0, x=-self.tau), ivt(i=0, x=self.tau)
        a2, b2 = ivt(i=1, x=-self.tau), ivt(i=1, x=self.tau)
        bound = 0
        bound += nu1.integrate_against_x(0, b1) - nu1.integrate_against_x(a1, 0)
        bound += nu2.integrate_against_x(0, b2) - nu2.integrate_against_x(a2, 0)
        return bound

    def df(self, t):
        return self.model1.df(t)

    def initialisation(self, product: Product, max_step_epsilon: float = None) -> None:
        self.times = product.times_grid()
        dt, nb = self.times.step, len(self.times)

        drift1 = self.model1.drift()
        drift2 = self.model2.drift()
        self._process_drift = np.array([[drift1], [drift2]])

        diffusion_coefficient_1 = self.model1.diffusion_coefficient()
        diffusion_coefficient_2 = self.model2.diffusion_coefficient()
        self.stddev_grid_1 = np.ones(nb-1)*diffusion_coefficient_1*np.sqrt(dt)
        self.stddev_grid_2 = np.ones(nb-1)*diffusion_coefficient_2*np.sqrt(dt)

    def one_simulation_cost(self, product) -> float:
        return 0

    def reset_one_simulation_cost(self) -> None:
        pass

    def pre_computation(self, mc_paths: int, product: Product) -> None:
        pass

    def simulate_one_path(self):
        # simulate the log-jump values
        log_jumps = self._simulate_jump_process()

        # simulate the diffusion part
        log_diff = self._simulate_diffusion()

        return StochasticJumpPath(self.times, log_diff, log_jumps)

    def _simulate_jump_process(self):
        tau = self.tau
        T = self.times[-1]

        # for i=1, 2 simulate N_i as Poisson(2*tau*T)
        poisson_2tau = Poisson(lam=2*tau*T).sample(size=2)
        N1, N2 = poisson_2tau[[0, 1]]

        rdm_uniform = Uniform()

        # simulate Ni random variable U1,...,U_{Ni}
        gamma_11 = tau*(2*rdm_uniform.sample(size=N1) - 1)
        gamma_22 = tau*(2*rdm_uniform.sample(size=N2) - 1)

        inverse = self.levy_copula_model.copula.inverse_conditional_distribution
        y1 = rdm_uniform.sample(size=N1)
        y2 = rdm_uniform.sample(size=N2)
        gamma_12 = inverse(gamma_11, y1)
        gamma_21 = inverse(gamma_22, y2)

        V = rdm_uniform.sample(size=max(N1, N2))*T

        partial_sums1 = np.zeros(shape=len(self.times))
        partial_sums2 = np.zeros(shape=len(self.times))

        ivt = self.levy_copula_model.inverse_tail_integral

        previous_set = np.flatnonzero(V < self.times[0])
        for k, (s, t) in enumerate(zip(self.times, self.times[1:])):
            # slice_st: indices i of V such as: s < V[i] <= t
            slice_st = np.setdiff1d(np.flatnonzero(V <= t), previous_set)
            previous_set = np.concatenate((slice_st, previous_set))

            if slice_st.size == 0:
                continue

            slice_1 = slice_st[np.flatnonzero(slice_st < N1)]
            slice_2 = slice_st[np.flatnonzero(slice_st < N2)]
            # Remark: by construction, all elements in gamma_11 and gamma_22 are smaller than tau, hence +1 below
            n1 = np.array(np.absolute(gamma_12[slice_1]) <= tau).astype(int) + 1
            n2 = np.array(np.absolute(gamma_21[slice_2]) <= tau).astype(int) + 1

            W1 = rdm_uniform.sample(size=n1.size)
            W2 = rdm_uniform.sample(size=n2.size)

            elements_n1 = slice_1[np.flatnonzero(np.multiply(n1, W1) <= 1)]
            elements_n2 = slice_2[np.flatnonzero(np.multiply(n2, W2) <= 1)]

            inverse_marginal1_gamma11 = np.sum(ivt(i=0, x=x) for x in gamma_11[elements_n1])
            inverse_marginal1_gamma21 = np.sum(ivt(i=0, x=x) for x in gamma_21[elements_n2])
            inverse_marginal2_gamma12 = np.sum(ivt(i=1, x=x) for x in gamma_12[elements_n1])
            inverse_marginal2_gamma22 = np.sum(ivt(i=1, x=x) for x in gamma_22[elements_n2])

            partial_sums1[k+1] = inverse_marginal1_gamma11 + inverse_marginal1_gamma21
            partial_sums2[k+1] = inverse_marginal2_gamma22 + inverse_marginal2_gamma12

        cum_sum1 = np.cumsum(partial_sums1)
        cum_sum2 = np.cumsum(partial_sums2)

        return np.array([cum_sum1, cum_sum2])

    def _simulate_diffusion(self):
        w1 = np.random.normal(size=self.stddev_grid_1.size)
        w2 = np.random.normal(size=self.stddev_grid_2.size)
        diff1 = np.concatenate(([0.0], +self.stddev_grid_1*w1))
        diff2 = np.concatenate(([0.0], +self.stddev_grid_2*w2))
        return np.cumsum(diff1), np.cumsum(diff2)
