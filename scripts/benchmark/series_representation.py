"""Replication of the results in "Simulation and option pricing in Lévy copula models" by Peter Tankov

The results are printed into a csv `results` sub-folder
"""

from collections import Callable
import csv
import os
from typing import Union

import pandas as pd
import numpy as np

from rpylib.distribution.levycopula import ClaytonCopula
from rpylib.grid.spatial import CTMCGridGeometric, CTMCUniformGrid, CTMCGrid
from rpylib.model.levymodel.levymodel import LevyModel
from rpylib.model.levycopulamodel import LevyCopulaModel
from rpylib.model.levymodel.purejump.variancegamma import VGParameters, ExponentialOfVarianceGammaModel
from rpylib.montecarlo.configuration import ConfigurationStandard
from rpylib.montecarlo.standard.engine import Engine
from rpylib.numerical.cosmethod import COSPricer
from rpylib.process.markovchain.markovchainlevycopula import MarkovChainLevyCopula
from rpylib.process.markovchain.markovchain import SamplingMethod
from rpylib.process.levycopulaseries import LevyCopula2dSeriesRepresentation
from rpylib.product.payoff import Vanilla, PayoffType
from rpylib.product.product import Product, ControlVariates
from rpylib.product.underlying import MaximumOfPerformances, Mean, NthSpot
from rpylib.tools.timer import timer
from rpylib.tools.system import create_folder


def pricing(configuration, process, product):
    engine = Engine(configuration=configuration, process=process)
    res = engine.price(product=product)
    price, mc_stddev = res.price(no_control_variates=True), res.mc_stddev(no_control_variates=True)
    price_with_cv, mc_stddev_with_cv = res.price(), res.mc_stddev()
    return price, mc_stddev, price_with_cv, mc_stddev_with_cv


@timer
def benchmark_series_representation(strikes: np.array, mc_paths: int, tau: float,
                                    ctmc_grid_function: Callable[[Union[LevyModel, LevyCopulaModel]], CTMCGrid],
                                    result_folder_name: str = None):
    # Variance Gamma models
    spot1, spot2, r, d = 1.0, 1.0, 0.03, 0.00
    params1 = VGParameters(sigma=0.30, nu=0.01, theta=-0.1)
    params2 = VGParameters(sigma=0.25, nu=0.01, theta=-0.1)
    model1 = ExponentialOfVarianceGammaModel(spot=spot1, r=r, d=d, parameters=params1)
    model2 = ExponentialOfVarianceGammaModel(spot=spot2, r=r, d=d, parameters=params2)

    maturity = 0.02
    asian_options = Product(payoff_underlying=Mean(),
                            payoff=Vanilla(strike=strikes*0.5*(spot1+spot2), payoff_type=PayoffType.CALL),
                            maturity=maturity)
    bestof_options = Product(payoff_underlying=MaximumOfPerformances([spot1, spot2]),
                             payoff=Vanilla(strike=strikes, payoff_type=PayoffType.CALL), maturity=maturity)
    all_options = [asian_options, bestof_options]

    # std: strong tail dependence
    # wtd: weak tail dependence
    copula_std, copula_wtd = ClaytonCopula(theta=10, eta=0.75), ClaytonCopula(theta=0.61, eta=0.99)
    levy_copula_model_std = LevyCopulaModel(models=[model1, model2], copula=copula_std)
    levy_copula_model_wtd = LevyCopulaModel(models=[model1, model2], copula=copula_wtd)
    levy_copula_models = {'std': levy_copula_model_std, 'wtd': levy_copula_model_wtd}

    # Monte-Carlo configuration
    call1 = Product(payoff_underlying=NthSpot(1), payoff=Vanilla(strike=spot1*strikes, payoff_type=PayoffType.CALL),
                    maturity=maturity)
    call2 = Product(payoff_underlying=NthSpot(2), payoff=Vanilla(strike=spot2*strikes, payoff_type=PayoffType.CALL),
                    maturity=maturity)
    price_calls = [COSPricer(model1).price(product=call1), COSPricer(model2).price(product=call2)]
    cv = ControlVariates([call1, call2], price_calls)
    configuration = ConfigurationStandard(mc_paths=mc_paths, control_variates=cv)
    method = SamplingMethod.BINARYSEARCHTREEADAPTED

    results_ctmc = [pricing(configuration, MarkovChainLevyCopula(model, ctmc_grid_function(model), method), options)
                    for _, model in levy_copula_models.items() for options in all_options]
    results_series = [pricing(configuration, LevyCopula2dSeriesRepresentation(model, tau=tau), options)
                      for _, model in levy_copula_models.items() for options in all_options]

    save_benchmark_series_results(strikes=strikes, levy_copula_models=levy_copula_models, all_options=all_options,
                                  results_ctmc=results_ctmc, results_series=results_series,
                                  result_folder_name=result_folder_name)


def create_result_files(strikes: list[float], products: list[str], result_folder: str, clear_file=False) -> None:
    for copula_type in ['std', 'wtd']:
        for product_type in products:
            file_name = copula_type + '_' + product_type
            file_path = os.path.join(result_folder, file_name + '.csv')

            if clear_file:
                f = open(file_path, "w+")
                f.close()

            if clear_file or not os.path.isfile(file_path) or os.stat(file_path).st_size == 0:
                # write header and strikes values
                with open(file_path, 'w') as f:
                    writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE, escapechar=' ')
                    header = 'strikes ctmcPrice ctmcMcStddev seriesPrice seriesMcStddev'
                    writer.writerow(header.split())
                    for strike in strikes:
                        writer.writerow([str(strike) + ',' + ',' + ',' + ','])


def dump_results(result_directory: str, process_type, copula_type: str, product_type: str, results):
    prices_no_cv, mc_stddevs_no_cv, prices_with_cv, mc_stddevs_with_cv = results

    file_name = copula_type + '_' + product_type
    file_path = os.path.join(result_directory, file_name + '.csv')

    df = pd.read_csv(file_path, sep=',')
    if process_type == 'ctmc':
        df['ctmcPrice'] = prices_with_cv
        df['ctmcMcStddev'] = mc_stddevs_with_cv
        df['ctmcPriceNoCv'] = prices_no_cv
        df['ctmcMcStddevNoCv'] = mc_stddevs_no_cv
    else:
        df['seriesPrice'] = prices_with_cv
        df['seriesMcStddev'] = mc_stddevs_with_cv
        df['seriesPriceNoCv'] = prices_no_cv
        df['seriesMcStddevNoCv'] = mc_stddevs_no_cv

    df.to_csv(file_path, index=False, sep=',', na_rep='nan')


def save_benchmark_series_results(strikes, levy_copula_models, all_options, results_ctmc, results_series,
                                  result_folder_name: str = None):
    # create and init the result folder
    result_folder_name_val = result_folder_name or os.path.join(os.getcwd(), 'results/series')
    create_folder(folder_name=result_folder_name_val)
    create_result_files(strikes=strikes, products=['asian', 'bestof'], result_folder=result_folder_name_val,
                        clear_file=True)

    # save results
    for i, key in enumerate(levy_copula_models):
        for j, _ in enumerate(all_options):
            product_type = 'asian' if j == 0 else 'bestof'
            dump_results(result_directory=result_folder_name_val, process_type="ctmc", copula_type=key,
                         product_type=product_type, results=results_ctmc[2 * i + j])
            dump_results(result_directory=result_folder_name_val, process_type="series", copula_type=key,
                         product_type=product_type, results=results_series[2 * i + j])


if __name__ == '__main__':
    my_strikes = np.array([0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05])

    # Recommended (starting) settings if you want to run this script on your personal computer as the
    # series representation is quite slow to simulate.
    # Choosing the `CTMCGridGeometric` also reduces the computing time (by a large factor) for the CTMC without
    # significantly impacting the accuracy.
    my_mc_paths = 100  # 50_000
    my_tau = 1_000     # 3_000

    def my_ctmc_grid_function(model):
        return CTMCGridGeometric(h=1e-6, model=model, nb_of_points_on_each_side=50)
        # return CTMCUniformGrid(h=1e-6, model=model)

    benchmark_series_representation(strikes=my_strikes, mc_paths=my_mc_paths, tau=my_tau,
                                    ctmc_grid_function=my_ctmc_grid_function)
