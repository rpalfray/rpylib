"""Replication of first to default CDS spread using the framework and pricing formula in
'A Structural Jump Threshold Framework for Credit Risk' by Pierre Garreau and Alec Kercheval

Results are written into a csv file.
"""

import os

import csv

from rpylib.distribution.sampling import SamplingMethod
from rpylib.grid.spatial import CTMCCredit, CTMCUniformGrid
from rpylib.model.utils import *
from rpylib.montecarlo.configuration import ConfigurationStandard
from rpylib.montecarlo.standard.engine import Engine
from rpylib.numerical.closedform.cflevymodel import CFLevyModel
from rpylib.numerical.closedform.cflevycopula import CFLevyCopulaModel
from rpylib.process.markovchain.markovchainlevycopula import MarkovChainLevyCopula
from rpylib.product.product import Product, ControlVariates
from rpylib.product.payoff import CDS
from rpylib.product.underlying import NthDefaultTimes, DefaultTimeNthUnderlying
from rpylib.tools.system import create_folder
from rpylib.tools.timer import timer

from scripts.benchmark.cds import helper_extract_result


def auxiliary_data(
    cds_spreads_bps, cf_pricer_copula, cf_pricers, recovery_rate, h0, maturity
):
    # compute threshold for each margin
    levels_as = [
        [
            cf_pricer.implied_cds_threshold(
                cds_spread=spread / 10_000, recovery_rate=recovery_rate, h0=h0
            )
            for cf_pricer in cf_pricers
        ]
        for spread in cds_spreads_bps
    ]
    survival_probabilities = [
        [
            cf_pricer.survival_probability(level_a=a, t=maturity)
            for a in list(zip(*levels_as))[k]
        ]
        for k, cf_pricer in enumerate(cf_pricers)
    ]
    for k, (prob, levels_as_margin) in enumerate(
        zip(survival_probabilities, list(zip(*levels_as))), start=1
    ):
        print("margin[" + str(k) + "]:")
        print(
            "levels a:               ",
            ", ".join("{:>8.4f}".format(level_a) for level_a in levels_as_margin),
        )
        print(
            "survival probabilities: ",
            ", ".join("{:>7.2f}%".format(100 * p) for p in prob),
        )
        print("")

    th_ftd_spread_bps = [
        10_000
        * cf_pricer_copula.first_to_default_par_spread(
            levels_a=levels_a, recovery_rate=recovery_rate
        )
        for levels_a in levels_as
    ]
    print(
        "theoretical 1s-to-default spreads (bps): ",
        ", ".join("{:>3.2f}".format(p) for p in th_ftd_spread_bps),
    )

    return levels_as, th_ftd_spread_bps, survival_probabilities


def helper_pricing_function(
    nth_index,
    h0,
    levy_copula_model,
    maturity,
    recovery_rate,
    models,
    notional,
    mc_paths,
    method,
):
    def cds(spread):
        return CDS(
            recovery_rate=recovery_rate,
            spread=spread,
            maturity=maturity,
            discounting=levy_copula_model.df,
        )

    def helper(
        spread,
        spread_margin,
        levels_a,
        symmetric_grid,
        credit_grid: bool,
        notional: float = 1,
    ):
        if credit_grid:
            grid = CTMCCredit(
                h=h0,
                level_a=levels_a,
                model=levy_copula_model,
                symmetric_grid=symmetric_grid,
            )
        else:
            grid = CTMCUniformGrid(h=h0, model=levy_copula_model)
        product = Product(
            payoff_underlying=NthDefaultTimes(default_levels=levels_a, index=nth_index),
            payoff=cds(spread=spread),
            maturity=maturity,
            notional=notional,
        )
        margins_cds = [
            Product(
                payoff_underlying=DefaultTimeNthUnderlying(
                    default_levels=levels_a, underlying_index=k
                ),
                payoff=cds(spread=spread_margin),
                maturity=maturity,
            )
            for k, model in enumerate(models, 1)
        ]
        margins_cds_prices = [
            0 for _ in enumerate(models)
        ]  # CDS are au pair for the margin spread
        cv = ControlVariates(margins_cds, margins_cds_prices)
        return grid, product, cv

    def pricing_function(spread, spread_margin, levels_a, credit_grid: bool = True):
        symmetric_grid = method != SamplingMethod.BINARYSEARCHTREEADAPTED
        grid, product, cv = helper(
            spread, spread_margin, levels_a, symmetric_grid, credit_grid, notional
        )
        process = MarkovChainLevyCopula(
            levy_copula_model=levy_copula_model, grid=grid, method=method
        )
        configuration = ConfigurationStandard(mc_paths=mc_paths, control_variates=cv)
        mc_engine = Engine(configuration=configuration, process=process)
        result = mc_engine.price(product=product)
        mc_price, mc_stddev = result.price(), result.mc_stddev()
        return mc_price, mc_stddev

    return pricing_function


@timer
def first_to_default_spread(
    name: str,
    copula,
    models,
    cds_spreads_bps: list[float],
    maturity: float,
    mc_paths: int,
    result_folder_name: str = None,
    use_credit_grid: bool = False,
):
    levy_copula_model = LevyCopulaModel(models=models, copula=copula)
    cf_pricer_copula = CFLevyCopulaModel(levy_copula_model=levy_copula_model)
    cf_pricers = [CFLevyModel(model=model) for model in models]

    h0 = 1e-6
    recovery_rate = 0.40
    notional = 1_000_000
    method = SamplingMethod.BINARYSEARCHTREEADAPTED

    levels_as, theoretical_spreads_bps, survival_probabilities = auxiliary_data(
        cds_spreads_bps, cf_pricer_copula, cf_pricers, recovery_rate, h0, maturity
    )
    pricing_function = helper_pricing_function(
        nth_index=1,
        h0=h0,
        levy_copula_model=levy_copula_model,
        maturity=maturity,
        recovery_rate=recovery_rate,
        models=models,
        notional=notional,
        mc_paths=mc_paths,
        method=method,
    )

    results = [
        pricing_function(
            spread / 10_000,
            spread_margin / 10_000,
            level_a,
            credit_grid=use_credit_grid,
        )
        for spread, spread_margin, level_a in zip(
            theoretical_spreads_bps, cds_spreads_bps, levels_as
        )
    ]

    is_bps, is_bps_stddev_m, is_bps_stddev_p, pvs = helper_extract_result(
        notional,
        cf_pricer_copula,
        recovery_rate,
        maturity,
        cds_spreads_bps,
        results,
        levels_as,
    )

    for spread, implied_spread, spread_stddev_m, spread_stddev_p, pv in zip(
        theoretical_spreads_bps, is_bps, is_bps_stddev_m, is_bps_stddev_p, pvs
    ):
        print(
            "th. spread: {:6.1f}(bps), implied spread: {:6.2f}(bps), stddev [{:>5.2f}/{:>5.2f}](bps), pv: {:6.3f} ".format(
                spread, implied_spread, spread_stddev_p, spread_stddev_m, pv
            )
        )

    save_benchmark_single_name_ftd_cds_results(
        name,
        cds_spreads_bps,
        theoretical_spreads_bps,
        levels_as,
        survival_probabilities,
        is_bps,
        is_bps_stddev_m,
        is_bps_stddev_p,
        result_folder_name,
    )


def save_benchmark_single_name_ftd_cds_results(
    name: str,
    cds_spreads_bps,
    theoretical_spreads_bps,
    levels_as,
    survival_probabilities,
    is_bps,
    is_bps_stddev_m,
    is_bps_stddev_p,
    result_folder_name: str = None,
):
    result_folder_name_val = result_folder_name or os.path.join(os.getcwd(), "results")
    create_folder(folder_name=result_folder_name_val)
    result_file_name = "first_to_default_" + name + ".csv"
    result_file_path = os.path.join(result_folder_name_val, result_file_name)

    with open(result_file_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_NONE, escapechar=" ")
        writer.writerow(
            "individualspread theoreticalspread levela survivalprobability impliedspread spreadstddevm "
            "spreadstddevp".split()
        )
        for s_margin, s, a, p, is_mc, is_m, is_p in zip(
            cds_spreads_bps,
            theoretical_spreads_bps,
            levels_as,
            list(zip(*survival_probabilities)),
            is_bps,
            is_bps_stddev_m,
            is_bps_stddev_p,
        ):
            levels_as_str = "[" + "/".join("{:6.3f}".format(ai) for ai in a) + "]"
            to_write = "{:6.0f} {:6.2f} {} {:6.4f} {:6.2f} {:>5.2f} {:>5.2f}".format(
                s_margin, s, levels_as_str, p[0], is_mc, is_p, is_m
            )
            writer.writerow(to_write.split())


if __name__ == "__main__":
    my_copula = create_clayton_copula()
    my_maturity = 1 / 2
    r = 0.02
    my_model1 = create_exponential_of_levy_model(ModelType.HEM)(
        r=r, spot=100, intensity=10, sigma=0.05
    )
    my_model2 = create_exponential_of_levy_model(ModelType.HEM)(
        r=r, spot=100, intensity=15, sigma=0.05
    )
    my_models = [my_model1, my_model2]
    my_mc_paths = 10_000
    my_cds_spreads_bps = [50, 150, 210]
    my_use_credit_grid = (
        True  # use credit grid for faster computation (without loss of accuracy)
    )

    first_to_default_spread(
        name="hem_hem",
        copula=my_copula,
        models=my_models,
        cds_spreads_bps=my_cds_spreads_bps,
        maturity=my_maturity,
        mc_paths=my_mc_paths,
        use_credit_grid=my_use_credit_grid,
    )
