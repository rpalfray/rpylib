"""Monte-Carlo engine.

    .. note::
        - the discounting is deterministic by design
        - the implementation supports multiprocessing
"""

import logging
import pathos.multiprocessing as mp

from tqdm import tqdm

from ..configuration import ConfigurationStandard
from ..path import create_path
from ..statistic.statistic import MCStatistics, create_mc_statistics
from ...model.levydrivensde.levydrivensde import LevyDrivenSDEModel
from ...process.process import Process
from ...product.product import Product, NoControlVariates


class Engine:
    """Standard Monte-Carlo engine"""
    def __init__(self, configuration: ConfigurationStandard, process: Process):
        """
        :param configuration: Monte-Carlo configuration
        :param process: stochastic process
        """
        self.configuration = configuration
        self.process = process
        self.path_manager = None
        self.statistics = None

    def initialisation(self, mc_paths: int, product: Product) -> None:
        """
        :param mc_paths: number of Monte-Carlo paths
        :param product: financial product to price
        """
        product.payoff_underlying.check_consistency(process_dimension=self.process.dimension())
        product.update(self.process.process_representation)
        if not isinstance(self.configuration.control_variates, NoControlVariates):
            for cv_product in self.configuration.control_variates.products:
                cv_product.update(self.process.process_representation)
        self.configuration.initialisation(product)
        self.process.initialisation(product)
        maturity = product.maturity

        # at the very end: initialise the path and pre-compute some stuff
        self.path_manager = create_path(self.configuration, self.process.deterministic_path)

        underlying_density = None
        if self.configuration.activate_spot_statistics:
            model = self.process.model
            if not isinstance(model, LevyDrivenSDEModel):
                if ((model.dimension() == 1 and not(hasattr(model, 'density') and callable(model.density)))
                        or (model.dimension() > 1 and not all(hasattr(m, 'density')
                                                              and callable(m.density) for m in model.models))):
                    logging.log(level=logging.WARNING, msg='No theoretical density for the underlying spot with this '
                                                           'model or it has not been implemented yet')
                else:
                    if model.dimension() == 1:
                        underlying_density = [model.density(t=maturity)]
                    else:
                        underlying_density = [m.density(t=maturity) for m in model.models]

        self.statistics = create_mc_statistics(mc_paths, underlying_density, self.configuration.control_variates,
                                               payoff_dimension=product.payoff.dimension(),
                                               process_representation=self.process.process_representation,
                                               activate_spot_statistics=self.configuration.activate_spot_statistics,
                                               spot_dimension=self.process.model.dimension())
        self.process.pre_computation(mc_paths, product)

    def price(self, product) -> MCStatistics:
        """Pricing of the product by the Monte-Carlo engine
        :param product: product to price
        """
        mc_paths = self.configuration.mc_paths
        self.initialisation(mc_paths, product)

        # Deterministic rates
        df = self.process.df(product.maturity)

        path_manager = self.path_manager
        path_manager.update(self.process.process_representation)  # this is needed for the embedded spot underlying
        # in the path
        statistics = self.statistics
        simulate_one_path = self.process.simulate_one_path
        cv = self.configuration.control_variates
        nb_of_processes = self.configuration.nb_of_processes

        # Monte-Carlo loop
        if nb_of_processes == 1:
            # single process version
            self.configuration.initialisation_seed()
            for iteration in range(mc_paths):
                simulated_path = simulate_one_path()
                # process the path: compute the payoff and discount it
                path_manager.set_to_path(simulated_path)
                path_manager.process(product, cv)
                path_manager.discount(df)
                # process the result: keep track of the statistics for the underlying and the payoff
                statistics.add(iteration, path_manager)
        else:
            # multiprocessor version
            def initializer():
                return self.configuration.initialisation_seed(nb_of_processes is None or nb_of_processes > 1)

            def callback(res):
                for it, mp_simulated_path in res:
                    # process the path: compute the payoff and discount it
                    path_manager.set_to_path(mp_simulated_path)
                    path_manager.process(product, cv)
                    path_manager.discount(df)
                    # process the result: keep track of the statistics for the underlying and the payoff
                    statistics.add(it, path_manager)

            def simulating_one_path(it):
                return it, simulate_one_path()

            with mp.Pool(processes=nb_of_processes, initializer=initializer) as pool:
                pool.map_async(simulating_one_path, tqdm(range(mc_paths)), callback=callback).get()

        # compute the adjustment if there are control variates
        cv.compute_coefficients(self.statistics)

        return self.statistics
