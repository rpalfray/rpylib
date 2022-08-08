"""Generate the variances vl and means ml for different levels in the Multilevel Monte-Carlo
for unidimensional Lévy models
"""

import pickle
from pathlib import Path

from scripts.mlmc.tools.utils import *

from rpylib.distribution.sampling import SamplingMethod
from rpylib.grid.spatial import CTMCGrid, CTMCUniformGrid
from rpylib.model.utils import *
from rpylib.montecarlo.configuration import (
    ConfigurationMultiLevel,
    compute_convergence_rates,
)
from rpylib.montecarlo.multilevel.engine import Engine
from rpylib.process.coupling.couplingmarkovchain import CouplingMarkovChain
from rpylib.product.payoff import Vanilla, PayoffType
from rpylib.product.product import Product
from rpylib.product.underlying import Spot
from rpylib.tools.timer import timer
from rpylib.tools.system import create_folder


def convergence_rate(
    model: LevyModel,
    product: Product,
    max_level: int,
    mc_paths: int,
    method: SamplingMethod,
    grid: CTMCGrid,
):
    coupling_process = CouplingMarkovChain(model=model, method=method, grid=grid)
    bg_index = model.levy_triplet.nu.blumenthal_getoor_index()
    cr = compute_convergence_rates(bg_index)
    convergence_criteria = None
    initial_level, maximum_level = 2, max_level
    initial_mc_paths = mc_paths
    configuration = ConfigurationMultiLevel(
        convergence_rates=cr,
        convergence_criteria=convergence_criteria,
        initial_level=initial_level,
        maximum_level=maximum_level,
        initial_mc_paths=initial_mc_paths,
    )
    mc_engine = Engine(configuration=configuration, coupling_process=coupling_process)
    return mc_engine.price_with_constant_mc_paths_and_level(product=product)


@timer
def coupling_variances(
    name,
    model,
    mc_paths: int,
    max_level: int,
    maturity: float,
    h0: float,
    result_folder_name: str = None,
):
    strike = atm_strike(model=model)
    payoff = Vanilla(strike=strike, payoff_type=PayoffType.PUT)
    product = Product(
        payoff_underlying=Spot(), payoff=payoff, maturity=maturity, notional=1.0
    )
    grid = CTMCUniformGrid(h=h0, model=model)

    method = SamplingMethod.BINARYSEARCHTREEADAPTED1D
    results = convergence_rate(model, product, max_level, mc_paths, method, grid)
    consistency_check = results.mlmc_results.consistency_check

    result_folder_name = result_folder_name or Path(
        Path(__file__).cwd().parent, "results/giles_convergence/1d"
    )
    file_path, data_path = helper_create_results_folder(
        name=name, result_folder_name=result_folder_name
    )

    # save the results
    beta = model.blumenthal_getoor_index()
    save_mlmc_coupling_convergence_results(file_path, results, consistency_check, beta)

    with open(data_path, "wb") as f:
        data = {"grid": grid, "model": model, "product": product, "method": method}
        pickle.dump(data, f)


if __name__ == "__main__":
    my_h0 = cases_data["h0"]
    my_maturity = cases_data["maturity"]
    # If you are running this code on your personal computer, you might want to take it easy with my_max_level
    # and my_mc_paths as the script might be quite time-consuming.
    # Note that you can replace the CTMCUniformGrid with a CTMCGridGeometric to speed it up.
    my_max_level = 5  # cases_data['max_level']
    my_mc_paths = 10_000  # cases_data['mc_paths']
    my_name = "cgmy02"
    my_model = exp_of_levy_models[my_name]

    coupling_variances(my_name, my_model, my_mc_paths, my_max_level, my_maturity, my_h0)
