"""Multilevel Monte-Carlo engine

     .. note::
        - the discounting is deterministic by design
        - the implementation supports multiprocessing
"""

import copy
import logging

import numpy as np
import pathos.multiprocessing as mp
from tqdm import tqdm

from ..configuration import ConfigurationMultiLevel
from ..path import create_path, MLMCPath
from ..statistic.statistic import create_mlmc_statistics, MLMCStatistics
from ...numerical.cosmethod import COSPricer
from ...process.coupling.couplingprocess import CouplingProcess
from ...process.markovchain.markovchainlevycopula import MarkovChainLevyCopula
from ...process.markovchain.markovchainsde import MarkovChainSDE
from ...product.product import Product


def helper_create_fun(this_cos_pricer, maturity):
    def fun(s):
        return this_cos_pricer.density(time=maturity, s=s)

    return fun


class Engine:
    """Multilevel Monte-Carlo engine"""

    def __init__(
        self, configuration: ConfigurationMultiLevel, coupling_process: CouplingProcess
    ):
        """The Multilevel Monte-Carlo relies on the simulation of the fine and coarse processes linked by a coupling.
        :param configuration: Multilevel Monte-Carlo configuration
        :param coupling_process: stochastic coupling process of the fine and coarse processes
        """
        self.configuration = configuration
        self.coupling_process = coupling_process
        self.path_managers: [MLMCPath] = []
        self.statistics: MLMCStatistics = None

    def initialisation(self, product: Product) -> None:
        """
        :param product: financial product to price
        """
        product.payoff_underlying.check_consistency(
            process_dimension=self.coupling_process.model.dimension_model()
        )
        maturity = product.maturity
        product.update(self.coupling_process.fine_process.process_representation)
        self.coupling_process.initialisation(product)
        self.configuration.initialisation(product)
        path_manager = create_path(
            self.configuration, self.coupling_process.fine_process.deterministic_path
        )
        self.path_managers.append(path_manager)
        self.coupling_process.pre_computation(
            mc_paths=self.configuration.initial_mc_paths, product=product
        )

        underlying_density = None
        if not (
            isinstance(self.coupling_process.fine_process, MarkovChainLevyCopula)
            or isinstance(self.coupling_process.fine_process, MarkovChainSDE)
        ):
            cos_pricer = COSPricer(self.coupling_process.model)
            underlying_density = [helper_create_fun(cos_pricer, maturity)]

        if isinstance(self.coupling_process.fine_process, MarkovChainLevyCopula):
            lcm = self.coupling_process.model
            cos_pricers = [COSPricer(model) for model in lcm.models]
            underlying_density = [
                helper_create_fun(cos_pricer, maturity) for cos_pricer in cos_pricers
            ]

        self.statistics = create_mlmc_statistics(
            mc_paths=self.configuration.initial_mc_paths,
            underlying_density=underlying_density,
            initial_level=self.configuration.initial_level,
            control_variates=self.configuration.control_variates,
            payoff_dimension=product.payoff.dimension(),
            process_representation=self.coupling_process.model.process_representation,
        )

    def compute_level_l(
        self,
        level: int,
        current_mc_paths: int,
        extra_mc_paths: int,
        coupling_process: CouplingProcess,
        product: Product,
        df: float,
        statistics: MLMCStatistics,
    ) -> None:
        """Run the Multilevel Monte-Carlo for the level l
        :param level: current level
        :param current_mc_paths: number of Monte-Carlo paths applied to the level l
        :param extra_mc_paths: extra number of paths to be applied to the level l
        :param coupling_process: coupling process defining the coupling between the fine and coarse processes
        :param product: product to price
        :param df: discount factor function
        :param statistics: statistics object
        """
        path_manager = self.path_managers[level]
        cv = self.configuration.control_variates

        if level == 0:
            simulation_path = coupling_process.simulate_one_path
            path_manager_process = path_manager.process_l0
        else:
            simulation_path = coupling_process.simulate_one_path_with_coupling
            path_manager_process = path_manager.process

        nb_of_processes = self.configuration.nb_of_processes

        if nb_of_processes == 1:
            # single process version
            self.configuration.initialisation_seed()
            for iteration in range(extra_mc_paths):
                simulated_path = simulation_path()
                path_manager.set_to_path(simulated_path)
                path_manager_process(product, cv)
                path_manager.discount(df)
                statistics.add(current_mc_paths + iteration, level, path_manager)
            # payoff -> [pl, plm], dp = pl - plm // pl: fine process, plm: coarse process
        else:
            description = "Computing level=" + str(level)

            def callback(res):
                for it, mp_simulated_path in res:
                    path_manager.set_to_path(mp_simulated_path)
                    path_manager_process(product, cv)
                    path_manager.discount(df)
                    statistics.add(current_mc_paths + it, level, path_manager)
                    # payoff -> [pl, plm], dp = pl - plm // pl: fine process, plm: coarse process

            def simulating_one_path(it):
                return it, simulation_path()

            def initializer():
                return self.configuration.initialisation_seed(
                    nb_of_processes is None or nb_of_processes > 1
                )

            with mp.Pool(processes=nb_of_processes, initializer=initializer) as pool:
                pool.map_async(
                    simulating_one_path,
                    tqdm(range(extra_mc_paths), desc=description, leave=True),
                    callback=callback,
                ).get()

        cv.compute_coefficients_mlmc(statistics.mc_statistics[level], level=level)

    def price(self, product: Product, rmse: float) -> MLMCStatistics:
        """Pricing of the product by the Multilevel Monte-Carlo engine
        :param product: product to price
        :param rmse: root-mean square error
        """
        self.initialisation(product)

        for path_manager in self.path_managers:
            path_manager.update(
                self.coupling_process.fine_process.process_representation
            )

        df = self.coupling_process.fine_process.df(product.maturity)

        N0 = self.configuration.initial_mc_paths
        L = self.configuration.initial_level
        level_max = self.configuration.maximum_level
        cr = self.configuration.convergence_rates
        alpha_0, beta_0, gamma_0 = cr.alpha, cr.beta, cr.gamma

        alpha = 0 if alpha_0 is None else alpha_0
        beta = 0 if beta_0 is None else beta_0
        gamma = 0 if gamma_0 is None else gamma_0

        ml_processes = [copy.deepcopy(self.coupling_process)]

        Nl = np.zeros(shape=L + 1, dtype=int)
        dNl = N0 * np.ones(shape=L + 1, dtype=int)
        sum_cost = np.zeros(shape=L + 1)
        statistics = self.statistics

        # use linear regression to estimate alpha, beta and gamma if not given
        def log2_regression(regress_to, max_val=0.5):
            mat = np.ones((L, 2))
            mat[:, 0] = range(1, L + 1)
            with np.errstate(
                divide="ignore"
            ):  # ignore 'divide by zero' warning as this is managed in 'res'
                x = np.linalg.lstsq(mat, np.log2(regress_to[1:]), rcond=None)[0]
            res = max(max_val, -x[0])
            return res

        while np.sum(dNl) > 0:
            for level in range(L + 1):
                if level > 0:
                    if (
                        len(ml_processes) < level + 1
                    ):  # create it by copying the process at level l-1 as starting point
                        logging.info("mlmc: adding level=" + str(level))
                        ml_process_level_l = copy.deepcopy(ml_processes[-1])
                        ml_process_level_l.next_level(
                            Nl[level], self.path_managers, product
                        )
                        ml_processes.append(ml_process_level_l)

                ml_processes[level].reset_one_simulation_cost()
                ml_processes[level].pre_computation(
                    mc_paths=dNl[level], product=product
                )
                self.compute_level_l(
                    level,
                    Nl[level],
                    dNl[level],
                    ml_processes[level],
                    product,
                    df,
                    statistics,
                )
                Nl[level] += dNl[level]
                sum_cost[level] += (
                    ml_processes[level].one_simulation_cost(product=product)
                    * dNl[level]
                )

            # compute absolute average and variance
            self.statistics.set_mlmc_results(Nl, sum_cost)
            ml = self.statistics.mlmc_results.ml
            vl = self.statistics.mlmc_results.vl
            cl = self.statistics.mlmc_results.cl

            # work-around for possible zero values
            for level in range(3, L + 1):
                ml[level] = np.maximum(ml[level], 0.5 * ml[level - 1] / 2**alpha)
                vl[level] = np.maximum(vl[level], 0.5 * vl[level - 1] / 2**beta)

            # if the convergence rates are not passed by the user, then they are estimated by linear regression
            if alpha_0 is None:
                alpha = log2_regression(ml)

            if beta_0 is None:
                beta = log2_regression(vl)

            if gamma_0 is None:
                gamma = log2_regression(cl)

            # set optimal number of additional paths
            Ns = self.configuration.convergence_criteria.compute_mc_paths(rmse, vl, cl)

            if Ns[0] > 10_000_000:
                logging.warning(
                    "Be patient, more than 10M of paths are used for the first level in the MLMC"
                )
                if Ns[0] > 50_000_000:
                    logging.error(
                        "The number of paths for the first level in the MLMC is more than 50M,"
                        "the rmse is probably to small in this case"
                    )

            dNl = np.maximum(0, Ns - Nl)

            if np.sum(dNl[dNl > 0.01 * Nl]) == 0:
                # test for convergence
                has_converged = self.configuration.convergence_criteria.criteria(
                    alpha, ml, rmse
                )
                if has_converged or L == level_max:
                    self.statistics.set_mlmc_results(Nl=Nl, sum_cost=sum_cost)
                    return self.statistics

                L += 1
                Nl = np.append(Nl, 1)
                vl = np.append(vl, vl[-1] / (2**beta))
                cl = np.append(cl, cl[-1] * (2**gamma))

                # optimal Nl
                Ns = self.configuration.convergence_criteria.compute_mc_paths(
                    rmse, vl, cl
                )
                dNl = np.maximum(0, Ns - Nl)
                sum_cost = np.append(sum_cost, 0.0)

                next_process = copy.deepcopy(ml_processes[-1])
                next_process.reset_one_simulation_cost()
                next_process.next_level(dNl[-1], self.path_managers, product=product)
                ml_processes.append(next_process)

            self.statistics.extend(Nl + dNl)

        logging.warning("Initial number of Monte-Carlo paths is probably too low")
        self.statistics.set_mlmc_results(Nl=Nl, sum_cost=sum_cost)
        return self.statistics

    def price_with_constant_mc_paths_and_level(self, product) -> MLMCStatistics:
        """Same as the :func:`price` function but with constant number of paths and a fixed number of levels
        :param product: product to price
        """
        mc_paths = self.configuration.initial_mc_paths
        max_level = self.configuration.maximum_level
        self.initialisation(product)
        for path_manager in self.path_managers:
            path_manager.update(
                self.coupling_process.fine_process.process_representation
            )

        df = self.coupling_process.fine_process.df(product.maturity)
        ml_process_level_l = copy.deepcopy(self.coupling_process)
        sum_cost = np.zeros(shape=max_level + 1)
        self.statistics.extend([mc_paths] * (max_level + 1))
        statistics = self.statistics

        for level in range(max_level + 1):
            if level > 0:
                ml_process_level_l.next_level(
                    mc_paths=mc_paths, path_managers=self.path_managers, product=product
                )

            self.compute_level_l(
                level=level,
                current_mc_paths=0,
                extra_mc_paths=mc_paths,
                coupling_process=ml_process_level_l,
                product=product,
                df=df,
                statistics=statistics,
            )
            sum_cost[level] = (
                ml_process_level_l.one_simulation_cost(product=product) * mc_paths
            )

        Nl = mc_paths * np.ones(max_level + 1)
        statistics.set_mlmc_results(Nl=Nl, sum_cost=sum_cost)

        return statistics
