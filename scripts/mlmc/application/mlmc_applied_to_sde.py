"""Application of the MLMC to a SDE-Lévy driven process with the CTMC scheme.
"""

from scripts.mlmc.tools.utils import *
from scripts.mlmc.application.mlmc_applied_to_copula import compute_max_level_copula

from rpylib.distribution.sampling import SamplingMethod
from rpylib.montecarlo.configuration import ConfigurationMultiLevel
from rpylib.montecarlo.configuration import compute_convergence_rates
from rpylib.montecarlo.multilevel.criteria import GilesConvergenceCriteria
from rpylib.montecarlo.multilevel.engine import Engine
from rpylib.process.coupling.couplingsde import CouplingSDE
from rpylib.montecarlo.statistic.statistic import MLMCStatistics
from rpylib.tools.timer import timer


def helper_coupling_sde(rmse: float, grid: CTMCGrid, model: LevyDrivenSDEModel, product: Product) -> MLMCStatistics:
    method = SamplingMethod.BinarySearchTreeAdapted
    maturity = product.maturity
    maximum_level = min(20, compute_max_level_copula(model.driver, h0=grid.h, maturity=maturity, rmse=rmse))

    cr = compute_convergence_rates(model.blumenthal_getoor_index())
    process = CouplingSDE(model=model, method=method, grid=grid)
    criteria = GilesConvergenceCriteria()
    configuration = ConfigurationMultiLevel(convergence_rates=cr, convergence_criteria=criteria,
                                            maximum_level=maximum_level)  # nb_of_processes=1
    mc_engine = Engine(configuration=configuration, coupling_process=process)
    res = mc_engine.price(product, rmse)

    return res


@timer
def coupling_sde_cost_and_levels(name: str, rmses: list[float]):
    root_path = Path(__file__).cwd().parent
    grid, model, product = helper_data(name, 'sde', root_path)
    beta = model.blumenthal_getoor_index()
    rmses = np.array(rmses)
    outputs = [helper_coupling_sde(rmse, grid, model, product) for rmse in rmses]
    root_path_results = Path(Path().cwd().parent, 'results/giles_applied/sde')
    save_mlmc_coupling_applied_results(rmses, outputs, root_path_results, name, beta)


if __name__ == '__main__':
    my_name = 'hem_cgmy02'
    my_rmses = [0.02, 0.01]

    # note that you must run the `mlmc_convergence_sde` script first with the same model
    # Same remark as for the `mlmc_convergence_sde` script: you might need to add `nb_of_processes=1` if you
    # are running this script in Windows.
    coupling_sde_cost_and_levels(name=my_name, rmses=my_rmses)
