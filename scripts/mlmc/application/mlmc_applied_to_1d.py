"""Application of the MLMC to a unidimensional Lévy process with the CTMC scheme.
"""

import math

from scripts.mlmc.tools.utils import *

from rpylib.distribution.sampling import SamplingMethod
from rpylib.grid.spatial import CTMCGrid
from rpylib.montecarlo.configuration import ConfigurationMultiLevel
from rpylib.montecarlo.configuration import compute_convergence_rates
from rpylib.montecarlo.multilevel.criteria import GilesConvergenceCriteria
from rpylib.montecarlo.multilevel.engine import Engine
from rpylib.process.coupling.couplingmarkovchain import CouplingMarkovChain
from rpylib.montecarlo.statistic.statistic import MLMCStatistics
from rpylib.tools.timer import timer


def compute_max_level(model: LevyModel, h0: float, dimension: int, maturity: float, rmse: float) -> int:
    mass = model.mass
    int_xx = model.levy_triplet.nu.integrate_against_xx
    bg = model.blumenthal_getoor_index()
    hl = [h0/2**level for level in range(5, 15)]

    def bound_fun(h):
        x1 = (h**2)*(mass(-np.inf, -h/2) + mass(h/2, np.inf))
        x2 = int_xx(-h, h)
        return max(x1, x2)

    log_h = np.log(np.array(hl))
    log_bounds = np.log(np.array([bound_fun(h) for h in hl]))
    logNcts = max(log_bounds - (2 - bg)*log_h)
    logDZero = np.log(8) + logNcts
    logDB = np.log(max(maturity, 1)*dimension) + logDZero
    two_minus_bg = 2 - bg

    res = math.floor((np.log(3) + two_minus_bg*np.log(h0) + logDB - 2*np.log(rmse))/(two_minus_bg*np.log(2)))
    return res


def helper_coupling(rmse: float, grid: CTMCGrid, model: LevyModel, product: Product) -> MLMCStatistics:
    method = SamplingMethod.BinarySearchTreeAdapted1D
    maturity = product.maturity
    maximum_level = min(20, compute_max_level(model, grid.h, 1, maturity, rmse))

    cr = compute_convergence_rates(model.blumenthal_getoor_index())
    process = CouplingMarkovChain(model=model, method=method, grid=grid)
    criteria = GilesConvergenceCriteria()
    configuration = ConfigurationMultiLevel(convergence_rates=cr, convergence_criteria=criteria,
                                            maximum_level=maximum_level)
    mc_engine = Engine(configuration=configuration, coupling_process=process)
    res = mc_engine.price(product, rmse)

    return res


@timer
def coupling_cost_and_levels(name: str, rmses: list[float]):
    root_path = Path(__file__).cwd().parent
    grid, model, product = helper_data(name, '1d', root_path)
    beta = model.blumenthal_getoor_index()
    rmses = np.array(rmses)
    output = [helper_coupling(rmse, grid, model, product) for rmse in rmses]
    root_path_results = Path(Path().cwd().parent, 'results/giles_applied/1d')
    save_mlmc_coupling_applied_results(rmses, output, root_path_results, name, beta)


if __name__ == '__main__':
    my_name = 'cgmy02'
    my_rmses = [0.2, 0.1]

    # note that you must run the `mlmc_convergence_1d` script first with the same model
    coupling_cost_and_levels(name=my_name, rmses=my_rmses)
