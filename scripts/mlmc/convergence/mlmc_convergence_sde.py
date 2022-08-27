"""Generate the variances vl and means ml for different levels in the Multilevel Monte-Carlo
for Lévy-driven SDE models where the driver is a 2d-Lévy copula model
"""

from scripts.mlmc.tools.utils import *

from rpylib.distribution.sampling import SamplingMethod
from rpylib.grid.spatial import CTMCGrid, CTMCUniformGrid
from rpylib.model.utils import *
from rpylib.montecarlo.configuration import (
    ConfigurationMultiLevel,
    compute_convergence_rates,
)
from rpylib.montecarlo.multilevel.engine import Engine
from rpylib.process.coupling.couplingsde import CouplingSDE
from rpylib.product.payoff import Swaption
from rpylib.product.product import Product
from rpylib.product.underlying import Libors
from rpylib.tools.timer import timer


def convergence_rate(
    model: LevyForwardModel,
    product: Product,
    max_level: int,
    mc_paths: int,
    method: SamplingMethod,
    grid: CTMCGrid,
):
    coupling_process = CouplingSDE(model=model, method=method, grid=grid)
    bg_index = model.blumenthal_getoor_index()
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
    )  # nb_of_processes=1
    mc_engine = Engine(configuration=configuration, coupling_process=coupling_process)
    return mc_engine.price_with_constant_mc_paths_and_level(product=product)


@timer
def coupling_variances_sde(
    name: str,
    model,
    mc_paths: int,
    h0: float,
    max_level: int,
    result_folder_name: str = None,
):
    product = Product(
        payoff_underlying=Libors(),
        payoff=Swaption(
            underlying_rates=model.x0, deltas=model.deltas, strike=np.average(model.x0)
        ),
        maturity=model.tenors[0],
        notional=100.0,
    )

    grid = CTMCUniformGrid(h=h0, model=model)
    if model.driver.dimension_model() == 1:
        method = SamplingMethod.BINARYSEARCHTREEADAPTED1D
    else:
        method = SamplingMethod.BINARYSEARCHTREEADAPTED

    results = convergence_rate(model, product, max_level, mc_paths, method, grid)
    consistency_check = results.mlmc_results.consistency_check

    result_folder_name = result_folder_name or Path(
        Path(__file__).cwd().parent, "results/giles_convergence/sde"
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
    # This script doesn't seem to work in Windows (but it should in Linux), and I suspect it is due to some global
    # objects in the code (multiprocessing `fork` works differently in Windows and Linux, see
    # https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods) but I haven't tried to figure
    # out where the problem is precisely (feel free to let me know!).
    # You will need to set `nb_of_processes=1` in the `ConfigurationMultiLevel` object
    # (in the `convergence_rate` function)

    my_h0 = cases_data["h0"]
    my_maturity = cases_data["maturity"]
    # If you are running this code on your personal computer, you might want to take it easy with my_max_level
    # and my_mc_paths as the script might be quite time-consuming.
    # Note that you can replace the CTMCUniformGrid with a CTMCGridGeometric to speed it up.
    my_max_level = 4  # cases_data['max_level']
    my_mc_paths = 1_000  # cases_data['mc_paths']

    my_name = "hem_cgmy02"
    models = [levy_models[n] for n in my_name.split("_")]
    driver = levy_models[my_name] if len(models) == 1 else models
    my_sde_driven_model = create_levy_forward_market_model_copula(driver=driver)
    # note that the copula function is set to the Clayton by default

    coupling_variances_sde(
        my_name, my_sde_driven_model, my_mc_paths, my_h0, my_max_level
    )
