"""Replication of a CDS spread using the framework and pricing formula in
'A Structural Jump Threshold Framework for Credit Risk' by Pierre Garreau and Alec Kercheval

The results are (partially) printed in the console and saved into a `results` sub-folder
"""

import os

import pandas as pd

from rpylib.grid.spatial import CTMCGrid, CTMCCredit, CTMCUniformGrid
from rpylib.model.utils import *
from rpylib.montecarlo.configuration import ConfigurationStandard
from rpylib.montecarlo.standard.engine import Engine
from rpylib.numerical.closedform.cflevymodel import CFLevyModel
from rpylib.process.markovchain.markovchain import MarkovChainProcess
from rpylib.process.markovchain.markovchain import SamplingMethod
from rpylib.product.product import Product
from rpylib.product.payoff import CDS
from rpylib.product.underlying import DefaultTime

from rpylib.tools.timer import timer
from rpylib.tools.system import create_folder


@timer
def cds_spread(name: str, model: LevyModel, maturity: float, mc_paths: int, cds_spreads_bps: list[float],
               ctmc_grid_function: Callable[[float, list[float]], CTMCGrid], result_folder_name: str = None):
    cf_pricer = CFLevyModel(model=model)
    recovery_rate = 0.4
    notional = 10_000
    h0 = 1e-6
    levels_as = [cf_pricer.implied_cds_threshold(cds_spread=spread/10_000, recovery_rate=recovery_rate, h0=h0)
                 for spread in cds_spreads_bps]
    survival_probabilities = [cf_pricer.survival_probability(level_a=level_a, t=maturity) for level_a in levels_as]
    print('levels a:               ', ', '.join('{:>8.4f}'.format(level_a) for level_a in levels_as))
    print('survival probabilities: ', ', '.join('{:>7.2f}%'.format(100*p) for p in survival_probabilities))

    configuration = ConfigurationStandard(mc_paths=mc_paths)

    def pricing_function(spread, levels_a):
        process = MarkovChainProcess(model=model, method=SamplingMethod.Inversion,
                                     grid=ctmc_grid_function(h0, levels_a))
        mc_engine = Engine(configuration=configuration, process=process)

        product = Product(payoff_underlying=DefaultTime(default_level=levels_a),
                          payoff=CDS(recovery_rate=recovery_rate, spread=spread, maturity=maturity,
                                     discounting=process.df),
                          maturity=maturity, notional=notional)

        result = mc_engine.price(product=product)
        mc_price_val, mc_stddev_val = result.price(), result.mc_stddev()
        return mc_price_val, mc_stddev_val

    results = [pricing_function(spread/10_000, level_a) for spread, level_a in zip(cds_spreads_bps, levels_as)]
    is_bps, is_bps_stddev_m, is_bps_stddev_p, pvs = \
        helper_extract_result(notional, cf_pricer, recovery_rate, maturity, cds_spreads_bps, results, levels_as)

    for spread, implied_spread, spread_stddev_m, spread_stddev_p, (mc_price, mc_stddev) \
            in zip(cds_spreads_bps, is_bps, is_bps_stddev_m, is_bps_stddev_p, results):
        print('spread: {:6.1f}(bps), implied spread: {:6.2f}(bps), stddev [{:>5.2f}/{:>5.2f}](bps), price: {:5.4f}, '
              'mc_stddev: {:5.4f}'.
              format(spread, implied_spread, spread_stddev_p, spread_stddev_m, mc_price, mc_stddev))

    save_benchmark_single_name_cds_results(name, cds_spreads_bps, levels_as, survival_probabilities,
                                           is_bps, is_bps_stddev_m, is_bps_stddev_p, result_folder_name)


def helper_extract_result(notional, cf_pricer, recovery_rate, maturity, cds_spreads_bps, results, levels_as):
    is_bps, is_bps_stddev_m, is_bps_stddev_p, pvs = [], [], [], []
    bps_scaling = 10_000
    z_value = 2.576  # 1.96  2.576

    for spread, (pv, pv_stddev), level_a in zip(cds_spreads_bps, results, levels_as):
        implied_spread = cf_pricer.implied_cds_spread(pv=pv/notional, level_a=level_a, recovery_rate=recovery_rate,
                                                      maturity=maturity)
        pstddev = cf_pricer.implied_cds_spread(pv=(pv+z_value*pv_stddev)/notional, level_a=level_a,
                                               recovery_rate=recovery_rate, maturity=maturity)
        mstddev = cf_pricer.implied_cds_spread(pv=(pv-z_value*pv_stddev)/notional, level_a=level_a,
                                               recovery_rate=recovery_rate, maturity=maturity)
        is_bps.append(bps_scaling*implied_spread)
        is_bps_stddev_m.append(bps_scaling*mstddev)
        is_bps_stddev_p.append(bps_scaling*pstddev)
        pvs.append(pv)

    return is_bps, is_bps_stddev_m, is_bps_stddev_p, pvs


def save_benchmark_single_name_cds_results(name: str, cds_spreads_bps, levels_as, survival_probabilities,
                                           is_bps, is_bps_stddev_m, is_bps_stddev_p, result_folder_name: str = None):
    # save results
    result_folder_name_val = result_folder_name or os.path.join(os.getcwd(), 'results')
    create_folder(folder_name=result_folder_name_val)
    result_file_name = 'cds_' + name + '.csv'
    result_file_path = os.path.join(result_folder_name_val, result_file_name)

    with open(result_file_path, 'a+', newline=''):
        df = pd.DataFrame(dtype=float)
        df['theoreticalspread'] = cds_spreads_bps
        df['levela'] = levels_as
        df['survivalprobability'] = survival_probabilities
        df['impliedspread'] = is_bps
        df['spreadstddevm'] = is_bps_stddev_p
        df['spreadstddevp'] = is_bps_stddev_m

        # note the rounding
        df.theoreticalspread = df.theoreticalspread.round(0)
        df.levela = df.levela.round(3)
        df.survivalprobability = df.survivalprobability.round(4)
        df.impliedspread = df.impliedspread.round(2)
        df.spreadstddevm = df.spreadstddevm.round(2)
        df.spreadstddevp = df.spreadstddevp.round(2)

        df.to_csv(result_file_path, index=False, sep=',', na_rep='nan')


if __name__ == '__main__':
    my_maturity = 3/12
    my_name = 'hem'
    my_model = create_exponential_of_levy_model(ModelType.HEM)(intensity=10)
    my_mc_paths = 10_000
    my_cds_spreads_bps = [150, 180]

    def my_ctmc_grid_function(my_h, my_levels_a):
        return CTMCCredit(h=my_h, level_a=my_levels_a, model=my_model)  # this version will be much faster without
        # any loss of accuracy
        # return CTMCUniformGrid(h=my_h, model=my_model)

    cds_spread(name=my_name, model=my_model, maturity=my_maturity, mc_paths=my_mc_paths,
               cds_spreads_bps=my_cds_spreads_bps, ctmc_grid_function=my_ctmc_grid_function)
