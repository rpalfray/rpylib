"""Store the simulations of an estimator (underlying, payoff) and add tools to describe its statistics:
moments, density, etc.

"""

import copy
from collections.abc import Callable
from enum import IntEnum

import matplotlib.pyplot as plt

from .tools import *
from ...process.process import ProcessRepresentation


class NoStatistic:
    """NoStatistic object: statistics for this variable are not needed"""

    def __init__(self):
        pass

    def add(self, simulation: int, variable: np.array) -> None:
        pass

    def extend(self, mc_path) -> None:
        pass

    def __repr__(self) -> str:
        return ''

    def plot_density(self):
        pass


class Density:
    """Define and plot the density function of a stochastic underlying modelled by a Lévy model

    """
    def __init__(self, label: str, process_representation: ProcessRepresentation):
        """
        :param label: label of the density plot
        :param process_representation: how the process is represented by the model (identity or log)
        """
        self.label = label
        self.density = None
        self.process_representation = process_representation

    def __call__(self, stats: np.array, title: str = '') -> None:
        """Plot the histogram of the data

        :param stats: statistics to be plotted
        :param title: title of the figure
        :return: a plot
        """
        dim = stats.shape[1]
        stats_to_use = [stats]

        if len(stats.shape) == 3:  # that's mlmc stats with fine and coarse values
            stats_to_use = [stats[..., PT.FP], stats[..., PT.CP]]
            dim *= 2

        fig, axs = plt.subplots(1, dim)
        for index, this_stats in enumerate(stats_to_use):
            for k, simulations in enumerate(this_stats.T):
                if dim == 1:
                    ax = axs
                else:
                    ax = axs[index*this_stats.shape[1] + k]
                n, bins, patches = ax.hist(simulations, bins='auto', density=True, facecolor='green', alpha=0.40)

                mean_val, stddev_val = mean(simulations), stddev(simulations)
                if stddev_val <= 1e-8:
                    # if the stddev is very small, we just take a fraction of the mean so that
                    # the min/max values are not equal to the mean
                    stddev_val = 0.1*mean_val
                axes = plt.gca()
                min_val = mean_val - 5*stddev_val
                max_val = mean_val + 5*stddev_val
                if self.process_representation == ProcessRepresentation.Log:
                    min_val = max(0, min_val)
                axes.set_xlim([min_val, max_val])

                if self.density:
                    y = self.density[k](bins)
                    ax.plot(bins, y, 'r--', alpha=0.60)

                ax.set_xlabel(self.label + '_' + str(k), fontsize='xx-small')

                # Tweak spacing to prevent clipping of the y-label
                plt.subplots_adjust(left=0.15)

        if title:
            plt.title(label=title)
        plt.show()

    def set_density(self, density):
        self.density = density


class Statistic:
    """Statistics object to manage the simulated variables and handle requests such as plotting their density"""
    def __init__(self, label: str, shape: tuple, mc_paths: int, process_representation: ProcessRepresentation):
        """
        :param label: label of the simulated variables
        :param shape: shape of the input data
        :param mc_paths: number of simulations in the Monte-Carlo engine
        """
        self.density_plot = Density(label=label, process_representation=process_representation)
        self.stats = np.empty(shape=(mc_paths, *shape))

    def plot_density(self):
        return self.density_plot(stats=self.stats)

    def add(self, simulation: int, variable: np.array) -> None:
        self.stats[simulation] = variable

    def extend(self, mc_path) -> None:
        """
        Extend the data to support extra number of Monte-Carlo paths
        :param mc_path: extra number of paths
        """
        if self.stats.shape[0] < mc_path:
            delta = mc_path - self.stats.shape[0]
            zero_padding = tuple((0, 0) for _ in range(len(self.stats.shape)-1))
            self.stats = np.pad(self.stats, ((0, delta), *zero_padding), "constant")
            # we just pad the first dimension which corresponds to the number of Monte-Carlo paths

    def __repr__(self) -> str:
        return 'Statistic'


class MCStatistics:
    """Statistics object when used with a standard Monte-Carlo engine"""
    def __init__(self, *, payoff_statistics: Statistic, spot_underlying_statistics: Statistic = NoStatistic(),
                 control_variates_statistics: Statistic = NoStatistic()):
        """
        :param payoff_statistics: statistics object handling the simulations of the payoff
        :param spot_underlying_statistics: statistics object handling the simulations of the spot underlying
        :param control_variates_statistics: statistics object handling the simulations of the control variates
        """
        self._payoff_statistics = payoff_statistics
        self._spot_underlying_statistics = spot_underlying_statistics
        self._control_variates_statistics = control_variates_statistics  # ->statistics of the control variates
        self._payoff_statistics_with_cv = copy.deepcopy(payoff_statistics)  # -> statistics of the adjusted payoff
        # with the control variates

    def add(self, simulation: int, path_manager: 'MCPath') -> None:
        self._spot_underlying_statistics.add(simulation, path_manager.spot_underlying)
        self._payoff_statistics.add(simulation, path_manager.payoff)
        self._control_variates_statistics.add(simulation, path_manager.payoff_control_variates)

    def extend(self, mc_path) -> None:
        self._spot_underlying_statistics.extend(mc_path)
        self._payoff_statistics.extend(mc_path)
        self._control_variates_statistics.extend(mc_path)
        self._payoff_statistics_with_cv.extend(mc_path)

    def __repr__(self) -> str:
        return self._payoff_statistics.__repr__()

    def _get_payoff_statistics(self, no_control_variates: bool = False) -> Statistic:
        if no_control_variates or isinstance(self._control_variates_statistics, NoStatistic):
            return self._payoff_statistics
        else:
            return self._payoff_statistics_with_cv

    def plot_spot_density(self):
        return self._spot_underlying_statistics.plot_density()

    def plot_payoff_density(self):
        return self._payoff_statistics.plot_density()

    def price(self, no_control_variates: bool = False) -> np.array:
        return self.get_mean(no_control_variates=no_control_variates)

    def mc_stddev(self, no_control_variates: bool = False) -> np.array:
        stats_obj = self._get_payoff_statistics(no_control_variates)
        res = mc_stddev(stats_obj.stats)
        return res[0] if res.size == 1 else res

    def get_mean(self, no_control_variates: bool = False) -> np.array:
        stats_obj = self._get_payoff_statistics(no_control_variates)
        res = mean(stats_obj.stats)
        return res[0] if res.size == 1 else res

    def get_variance(self, no_control_variates: bool = False) -> np.array:
        stats_obj = self._get_payoff_statistics(no_control_variates)
        res = mean(stats_obj.stats)
        return res[0] if res.size == 1 else res


class MLMCResults:
    """Handle the output of the Multilevel Monte-Carlo"""

    def __init__(self, Nl: np.array, sum_cost: np.array, all_pl_fine: list, all_pl_coarse: list):
        """
        :param Nl: number of Monte-Carlo paths for each level l
        :param sum_cost: array of the cost for each level l
        :param all_pl_fine: correction terms for the `fine` payoff for each level l
        :param all_pl_coarse: correction terms for the `coarse` payoff for each level l
        """
        dp = [pl_fine - pl_coarse for pl_fine, pl_coarse in zip(all_pl_fine, all_pl_coarse)]
        self._ncms_dp = NonCenteredMoments(dp)
        self._ncms_fine = NonCenteredMoments(all_pl_fine)
        self._sum_cost = sum_cost
        self.Nl = Nl

    @property
    def mean_level_l(self):
        val = self._ncms_fine.ncm_first
        return val

    @property
    def var_level_l(self):
        val = self._ncms_fine.ncm_second - self.mean_level_l**2
        return val

    @property
    def kurtosis(self):
        nsum1 = self._ncms_dp.ncm_first
        nsum2 = self._ncms_dp.ncm_second
        nsum3 = self._ncms_dp.ncm_third
        nsum4 = self._ncms_dp.ncm_fourth
        val = (nsum4 - 4*nsum3*nsum1 + 6*nsum2*nsum1**2 - 3*nsum1 ** 4)/(np.maximum(1, nsum2 - nsum1**2))**2
        return val

    @property
    def ml(self):
        """Mean of the correction terms |Pl - P{l-1}|  for the level l"""
        val = np.absolute(self._ncms_dp.ncm_first)
        return val

    @property
    def vl(self):
        """Variance of the correction terms |Pl - P{l-1}| for the level l"""
        val = np.maximum(0., self._ncms_dp.ncm_second - self._ncms_dp.ncm_first**2)
        return val

    @property
    def cl(self):
        """Algorithm cost associated to the level l"""
        val = self._sum_cost/self.Nl
        return val

    @property
    def cost(self):
        """total cost of the Multilevel Monte-Carlo algorithm"""
        val = np.sum(self._sum_cost)
        return val

    @property
    def consistency_check(self):
        val = (self.ml[1:] - self.mean_level_l[1:] + self.mean_level_l[:1]) / \
              (3*(np.sqrt(self.vl[1:]) + np.sqrt(self.var_level_l[1:]) + np.sqrt(self.var_level_l[:1])))
        return val


class PT(IntEnum):
    FP = 0   # fine process position
    CP = 1   # coarse process position


class MLMCStatistics:
    """Statistics object when used with a Multilevel Monte-Carlo engine"""
    def __init__(self, mc_statistics: list[MCStatistics], create_statistic):
        """
        :param mc_statistics: list of the statistics object for each level l
        :param create_statistic: helper function to create individual statistics
        """
        self.mc_statistics = mc_statistics
        self.create_statistic = create_statistic
        self.mlmc_results: MLMCResults = None

    def add(self, simulation: int, level: int, path_manager: 'MCPath') -> None:
        self.mc_statistics[level].add(simulation, path_manager)

    def set_mlmc_results(self, Nl: np.array, sum_cost: np.array):
        all_pl_fine = [self.simulation_payoff_with_fine_process(level=level) for level in range(len(Nl))]
        all_pl_coarse = [self.simulation_payoff_with_coarse_process(level=level) for level in range(len(Nl))]
        self.mlmc_results = MLMCResults(Nl=Nl, sum_cost=sum_cost, all_pl_fine=all_pl_fine, all_pl_coarse=all_pl_coarse)

    def add_mc_statistic(self):
        self.mc_statistics.append(self.create_statistic())

    def extend(self, mc_paths):
        while len(self.mc_statistics) < len(mc_paths):
            self.add_mc_statistic()

        for level, mc_stat in enumerate(self.mc_statistics):
            mc_stat.extend(mc_paths[level])

    def plot_spot_density(self, level: int = None):
        if level is None:  # then plot for all the levels
            for stats in self.mc_statistics:
                stats._spot_underlying_statistics.plot_density()
        else:
            return self.mc_statistics[level]._spot_underlying_statistics.plot_density()

    def _get_payoff_statistics(self, level: int, no_control_variates: bool = False) -> Statistic:
        stats_level_l = self.mc_statistics[level]
        if no_control_variates or isinstance(stats_level_l._control_variates_statistics, NoStatistic):
            return stats_level_l._payoff_statistics
        else:
            return stats_level_l._payoff_statistics_with_cv

    def price(self, no_control_variates: bool = False) -> float:
        res = 0
        for level, mc_stat in enumerate(self.mc_statistics):
            stats_obj = self._get_payoff_statistics(level=level, no_control_variates=no_control_variates)
            mean_val = mean(stats_obj.stats)
            res += mean_val[0, PT.FP] - mean_val[0, PT.CP]
        return res

    def mc_stddev(self, no_control_variates: bool = False) -> float:
        res = 0
        for level, mc_stat in enumerate(self.mc_statistics):
            pl_fine = self.simulation_payoff_with_fine_process(level=level, no_control_variates=no_control_variates)
            pl_coarse = self.simulation_payoff_with_coarse_process(level=level, no_control_variates=no_control_variates)
            stddev_val = mc_stddev(pl_fine - pl_coarse)
            res += stddev_val
        return res

    def simulation_payoff_with_fine_process(self, level, start: int = None, end: int = None,
                                            no_control_variates: bool = False) -> np.array:
        stats = self._get_payoff_statistics(level=level, no_control_variates=no_control_variates).stats
        return stats[(slice(start, end), 0, PT.FP)]

    def simulation_payoff_with_coarse_process(self, level, start: int = None, end: int = None,
                                              no_control_variates: bool = False) -> np.array:
        stats = self._get_payoff_statistics(level=level, no_control_variates=no_control_variates).stats
        return stats[(slice(start, end), 0, PT.CP)]


def create_mc_statistics(mc_paths: int, underlying_density: [Callable[[float], float]],
                         control_variates: 'ControlVariates', payoff_dimension: int,
                         process_representation: ProcessRepresentation, activate_spot_statistics: bool,
                         spot_dimension: int) -> MCStatistics:
    """Factory method which returns the MCStatistics object

    :param mc_paths: number of Monte-Carlo paths
    :param underlying_density: theoretical underlying density
    :param control_variates: control variates object
    :param payoff_dimension: dimension of the payoff (that is the number of underlyings passed to the payoff function)
    :param process_representation: process representation in the model
    :param activate_spot_statistics: if True, compute and save the simulated values of the modelled underlying process
    :param spot_dimension: dimension of the modelled underlying process
    """
    from ...product.product import NoControlVariates

    # always create the payoff statistics
    payoff_statistics = Statistic('payoff', shape=(payoff_dimension,), mc_paths=mc_paths,
                                  process_representation=process_representation)

    spot_underlying_statistics = NoStatistic()
    if activate_spot_statistics:
        spot_underlying_statistics = Statistic('spot', shape=(spot_dimension,), mc_paths=mc_paths,
                                               process_representation=process_representation)
    if underlying_density is not None:
        spot_underlying_statistics.density_plot.set_density(underlying_density)

    if isinstance(control_variates, NoControlVariates):
        control_variates_statistics = NoStatistic()
    else:
        control_variates_statistics = Statistic('CV', shape=(len(control_variates.products), payoff_dimension),
                                                mc_paths=mc_paths, process_representation=process_representation)

    return MCStatistics(payoff_statistics=payoff_statistics, spot_underlying_statistics=spot_underlying_statistics,
                        control_variates_statistics=control_variates_statistics)


def create_mlmc_statistics(mc_paths: int, underlying_density: [Callable[[float], float]], initial_level: int,
                           control_variates: 'ControlVariates', payoff_dimension: int,
                           process_representation: ProcessRepresentation) -> MLMCStatistics:
    """Factory method which returns the MLMCStatistics object

    :param mc_paths: number of Monte-Carlo paths
    :param underlying_density: theoretical underlying density
    :param initial_level: initial number of levels
    :param control_variates: control variates object
    :param payoff_dimension: dimension of the payoff (that is the number of underlyings passed to the payoff function)
    :param process_representation: process representation in the model
    """
    from ...product.product import NoControlVariates
    payoff_stat = Statistic('payoff', shape=(payoff_dimension, 2), mc_paths=mc_paths,
                            process_representation=process_representation)  # coarse payoff is set to 0 for convenience

    spot_underlying_stat = NoStatistic()
    if underlying_density is not None:
        spot_dimension = len(underlying_density)
        spot_underlying_stat = Statistic('spot', shape=(spot_dimension,), mc_paths=mc_paths,
                                         process_representation=process_representation)
        spot_underlying_stat.density_plot.set_density(underlying_density)

    if isinstance(control_variates, NoControlVariates):
        control_variate_statistics0 = NoStatistic()
    else:
        control_variate_statistics0 = Statistic('CV', shape=(len(control_variates.products), payoff_dimension),
                                                mc_paths=mc_paths, process_representation=process_representation)
    mc_stat0 = MCStatistics(payoff_statistics=payoff_stat, spot_underlying_statistics=spot_underlying_stat,
                            control_variates_statistics=control_variate_statistics0)

    def create_statistic(nb_of_paths=0):
        # coarse and fine simulations for each spot and control variates underlying
        # -> for each simulation (row), each underlying (col) has a fine value and a coarse value
        # that is to say the last dimension always corresponds to the fine/coarse values
        c_payoff_stat = Statistic('payoff', shape=(payoff_dimension, 2), mc_paths=nb_of_paths,
                                  process_representation=process_representation)

        c_spot_underlying_stat = NoStatistic()
        if underlying_density is not None:
            c_spot_underlying_stat = Statistic('spot', shape=(spot_dimension, 2), mc_paths=nb_of_paths,
                                               process_representation=process_representation)
            # one needs two functions for both the coarse and fine process
            c_spot_underlying_stat.density_plot.set_density(underlying_density + underlying_density)

        if isinstance(control_variates, NoControlVariates):
            control_variate_statistics = NoStatistic()
        else:
            control_variate_statistics = Statistic('CV', shape=(len(control_variates.products), payoff_dimension, 2),
                                                   mc_paths=nb_of_paths, process_representation=process_representation)

        return MCStatistics(payoff_statistics=c_payoff_stat, spot_underlying_statistics=c_spot_underlying_stat,
                            control_variates_statistics=copy.deepcopy(control_variate_statistics))

    mc_stats = [mc_stat0] + [create_statistic(mc_paths) for _ in range(1, initial_level+1)]

    return MLMCStatistics(mc_stats, create_statistic)
