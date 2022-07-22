"""Generate the variances vl and means ml for different levels in the Multilevel Monte-Carlo
for Lévy copula models
"""

import pickle
from pathlib import Path

from scripts.mlmc.tools.utils import *

from rpylib.distribution.sampling import SamplingMethod
from rpylib.grid.spatial import CTMCGrid, CTMCUniformGrid
from rpylib.model.levycopulamodel import LevyCopulaModel
from rpylib.montecarlo.configuration import ConfigurationMultiLevel, compute_convergence_rates
from rpylib.montecarlo.multilevel.engine import Engine
from rpylib.process.coupling.couplinglevycopula import CouplingProcessLevyCopula
from rpylib.product.payoff import Vanilla, PayoffType
from rpylib.product.product import Product
from rpylib.product.underlying import MaximumOfPerformances
from rpylib.tools.timer import timer
from rpylib.tools.system import create_folder


def convergence_rate_copula(levy_copula_model: LevyCopulaModel, product: Product, max_level: int, mc_paths: int,
                            method: SamplingMethod, grid: CTMCGrid):

    coupling_process = CouplingProcessLevyCopula(levy_copula_model=levy_copula_model, grid=grid, method=method)
    bg_index = levy_copula_model.blumenthal_getoor_index()
    cr = compute_convergence_rates(bg_index)
    convergence_criteria = None
    initial_mc_paths = mc_paths
    configuration = ConfigurationMultiLevel(convergence_rates=cr, convergence_criteria=convergence_criteria,
                                            initial_level=2, maximum_level=max_level, initial_mc_paths=initial_mc_paths)
    mc_engine = Engine(configuration=configuration, coupling_process=coupling_process)
    results = mc_engine.price_with_constant_mc_paths_and_level(product=product)
    return results


@timer
def coupling_variances_copula(name: str, levy_copula_model: LevyCopulaModel, mc_paths: int, max_level: int,
                              maturity: float, result_folder_name: str = None, h0: float = None):
    h0 = h0 or cases_data['h0']
    spots = [model.spot for model in levy_copula_model.models]
    product = Product(payoff_underlying=MaximumOfPerformances(spots),
                      payoff=Vanilla(strike=1.00, payoff_type=PayoffType.CALL), maturity=maturity, notional=100)

    method = SamplingMethod.BinarySearchTreeAdapted
    grid = CTMCUniformGrid(h=h0, model=levy_copula_model)

    results = convergence_rate_copula(levy_copula_model, product, max_level, mc_paths, method, grid)
    consistency_check = results.mlmc_results.consistency_check

    result_folder_name = result_folder_name or Path(Path(__file__).cwd().parent, 'results/giles_convergence/copulas')
    file_path, data_path = helper_create_results_folder(name=name, result_folder_name=result_folder_name)

    # save the results
    beta = levy_copula_model.blumenthal_getoor_index()
    save_mlmc_coupling_variances_results(file_path, results, consistency_check, beta)

    with open(data_path, 'wb') as f:
        data = {'grid': grid, 'model': levy_copula_model, 'product': product, 'method': method}
        pickle.dump(data, f)


if __name__ == '__main__':
    my_h0 = cases_data['h0']
    my_maturity = cases_data['maturity']
    # If you are running this code on your personal computer, you might want to take it easy with my_max_level
    # and my_mc_paths as the script might be quite time-consuming.
    # Note that you can replace the CTMCUniformGrid with a CTMCGridGeometric to speed it up.
    my_max_level = 4  # cases_data['max_level']
    my_mc_paths = 1_000  # cases_data['mc_paths']

    my_name = 'hem_cgmy02'
    models = [calibrated_models[n] for n in my_name.split('_')]
    copula = create_clayton_copula()
    my_levy_copula_model = create_levy_copula_model(models=models, copula=copula)

    coupling_variances_copula(name=my_name, levy_copula_model=my_levy_copula_model, mc_paths=my_mc_paths,
                              max_level=my_max_level, h0=my_h0, maturity=my_maturity)
