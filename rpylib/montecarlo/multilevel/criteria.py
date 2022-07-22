"""Convergence criteria for the Multi-level Monte-Carlo
"""

from collections.abc import Callable

import numpy as np


class ConvergenceCriteria:
    """Characterisation of the criteria convergence and the update of the number of paths after each pass"""
    def __init__(self, criteria: Callable[[float, np.array, float], bool],
                 compute_mc_paths: Callable[[float, np.array, np.array], np.array]):
        """
        :param criteria: criteria convergence function
        :param compute_mc_paths: function updating the number of paths for each level
        """
        self.criteria = criteria
        self.compute_mc_paths = compute_mc_paths


def compute_mc_paths_giles(rmse: float, vl: np.array, cl: np.array) -> np.array:
    """Same as in Giles papers
    :param rmse: root mean square error
    :param vl: estimated variance of the correction terms |Pl - P{l-1}|
    :param cl: cost of each level l
    :return: the updated number of Monte-Carlo paths for each level l
    """
    theta = 0.25
    cl_zerocost = cl.copy()
    cl_zerocost[cl_zerocost == 0] = 1e30  # to avoid potential division by 0 in the following line
    return np.ceil(np.sqrt(vl/cl_zerocost)*np.sum(np.sqrt(vl*cl))/((1 - theta)*rmse**2)).astype(int)


def criteria_giles(alpha: float, ml: np.array, rmse: float) -> bool:
    """Same convergence criteria as in Giles papers

    :param alpha: weak convergence rate
    :param ml: estimated mean of the correction terms |Pl - P{l-1}|
    :param rmse: root mean square error
    :return: true if the convergence criteria has been met
    """
    rem = max(ml[-1], ml[-2]/2**alpha, ml[-3]/2**(2*alpha))/(2**alpha - 1)
    return rem <= rmse/np.sqrt(2)


def criteria_run_to_maximum_level(alpha: float, ml: np.array, rmse: float) -> bool:
    """Run the Multilevel Monte-Carlo until the last level
    :return: always false so that the algorithm runs until the maximum level
    """
    return False


class GilesConvergenceCriteria(ConvergenceCriteria):

    def __init__(self):
        super().__init__(criteria=criteria_giles, compute_mc_paths=compute_mc_paths_giles)
