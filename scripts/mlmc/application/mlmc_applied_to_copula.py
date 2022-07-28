"""Application of the MLMC to a unidimensional Lévy process with the CTMC scheme.
"""

from itertools import product
import math

from scripts.mlmc.tools.utils import *

from rpylib.distribution.sampling import SamplingMethod
from rpylib.model.levycopulamodel import LevyCopulaModel
from rpylib.montecarlo.configuration import ConfigurationMultiLevel
from rpylib.montecarlo.configuration import compute_convergence_rates
from rpylib.montecarlo.multilevel.criteria import GilesConvergenceCriteria
from rpylib.montecarlo.multilevel.engine import Engine
from rpylib.montecarlo.statistic.statistic import MLMCStatistics
from rpylib.process.markovchain.markovchainlevycopula import vol_adjustment_ij
from rpylib.process.coupling.couplinglevycopula import CouplingProcessLevyCopula
from rpylib.tools.timer import timer


def compute_max_level_copula(levy_copula_model: LevyCopulaModel, h0: float, maturity: float, rmse: float) -> int:
    dimension = levy_copula_model.dimension()
    bg = levy_copula_model.blumenthal_getoor_index()
    mass = levy_copula_model.mass

    def int_xx(h):
        res = sum(vol_adjustment_ij(i, i, h=2*h, levy_model=levy_copula_model) for i in range(dimension))
        return res

    def fun1(h):
        a = [-h/2]*dimension
        b = [+h/2]*dimension
        intervals = [[[h_l, h_r], [-np.inf, h_l], [h_r, np.inf]] for h_l, h_r in zip(a, b)]
        cartesian_product = product(*intervals)
        # discard first set which is [h_l1, h_r1]x[h_l2, h_r2]x...x[h_ln, h_rn]
        # we are calculating the measure on the complement of this very set
        next(cartesian_product)

        res = 0
        for c_set in cartesian_product:
            a, b = zip(*c_set)
            res += mass(a=a, b=b)
        return res

    def bound_fun(h):
        x1 = (dimension*h**2)*fun1(h)
        x2 = int_xx(h)
        return max(x1, x2)

    hl = [h0/2**level for level in range(5, 15)]
    log_h = np.log(np.array(hl))
    log_bounds = np.log(np.array([bound_fun(h) for h in hl]))
    logNcts = max(log_bounds - (2 - bg)*log_h)
    logDZero = np.log(8) + logNcts
    logDB = np.log(max(maturity, 1)*dimension) + logDZero
    two_minus_bg = 2 - bg

    val = math.floor((np.log(3) + two_minus_bg*np.log(h0) + logDB - 2*np.log(rmse))/(two_minus_bg*np.log(2)))
    return val


def helper_coupling_copula(rmse: float, grid: CTMCGrid, levy_copula_model: LevyCopulaModel, product: Product) \
        -> MLMCStatistics:
    method = SamplingMethod.BinarySearchTreeAdapted
    maximum_level = compute_max_level_copula(levy_copula_model=levy_copula_model, h0=grid.h, maturity=product.maturity,
                                             rmse=rmse)
    maximum_level = min(20, maximum_level)

    cr = compute_convergence_rates(levy_copula_model.blumenthal_getoor_index())
    coupling_process = CouplingProcessLevyCopula(levy_copula_model=levy_copula_model, grid=grid, method=method)
    criteria = GilesConvergenceCriteria()
    configuration = ConfigurationMultiLevel(convergence_rates=cr, convergence_criteria=criteria,
                                            maximum_level=maximum_level)
    mc_engine = Engine(configuration=configuration, coupling_process=coupling_process)
    result = mc_engine.price(product, rmse)

    return result


@timer
def coupling_copula_cost_and_levels(name: str, rmses: list[float]) -> None:
    root_path = Path(__file__).cwd().parent
    grid, model, product = helper_data(name, 'copulas', root_path)
    beta = model.blumenthal_getoor_index()
    rmses = np.array(rmses)
    outputs = [helper_coupling_copula(rmse, grid, model, product) for rmse in rmses]
    root_path_results = Path(Path().cwd().parent, 'results/giles_applied/copulas')
    save_mlmc_coupling_convergence_results(rmses, outputs, root_path_results, name, beta)


if __name__ == '__main__':
    my_name = 'hem_cgmy02'
    my_rmses = [0.2, 0.1]

    # note that you must run the `mlmc_convergence_copula` script first with the same model
    coupling_copula_cost_and_levels(name=my_name, rmses=my_rmses)
