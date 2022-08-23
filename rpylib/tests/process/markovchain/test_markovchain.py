"""Testing the Markov-Chain process implementation
"""

import numpy as np

from rpylib.distribution.sampling import SamplingMethod

# from rpylib.montecarlo.configuration import ConfigurationStandard
# from rpylib.montecarlo.path import StochasticJumpPath
# from rpylib.montecarlo.standard.engine import Engine
from rpylib.process.markovchain.markovchain import vol_adjustment, MarkovChainProcess
from rpylib.tests.conftest import spatial_grid


def test_bs_vol_adjustment(bs_model):
    value = vol_adjustment(model=bs_model, h=0.01)
    assert np.isclose(value, 0.0)


def test_cgmy_vol_adjustment(cgmy_model):
    value = vol_adjustment(model=cgmy_model, h=0.01)
    expected_value = 0.030227654765023124
    assert np.isclose(value, expected_value)


def test_hem_vol_adjustment(hem_model):
    value = vol_adjustment(model=hem_model, h=0.01)
    assert np.isclose(value, 0.0)


def checks_process_markov_chain(
    model,
    product,
    expected_ioj,
    expected_vol_adj,
    expected_mc_price,
    expected_mc_stddev,
):
    grid = spatial_grid(model=model)
    mcp = MarkovChainProcess(model=model, grid=grid, method=SamplingMethod.ALIAS)
    ioj = mcp.intensity_of_jumps
    vol_adj = vol_adjustment(model=model, h=grid.h)

    # mc_configuration = ConfigurationStandard(mc_paths=5, nb_of_processes=1, seed=123456)
    # mc_engine = Engine(configuration=mc_configuration, process=mcp)
    # result = mc_engine.price(product=product)
    # mc_price, mc_stddev = result.price(), result.mc_stddev()

    assert np.isclose(ioj, expected_ioj)
    assert np.isclose(vol_adj, expected_vol_adj)
    # FIXME: difference between release/debug
    # assert(np.allclose(mc_price, expected_mc_price))
    # assert(np.allclose(mc_stddev, expected_mc_stddev))


def test_hem_process_markov_chain_testing_value(hem_model, forward):
    expected_ioj = 4.28495221390975
    expected_vol_adj = 0.0
    expected_mc_price = -0.9866148102370456
    expected_mc_stddev = 1.6269584124784136
    checks_process_markov_chain(
        hem_model,
        forward,
        expected_ioj,
        expected_vol_adj,
        expected_mc_price,
        expected_mc_stddev,
    )


def test_cgmy_process_markov_chain_testing_value(cgmy_model, forward):
    expected_ioj = 26.166020822055167
    expected_vol_adj = 0.030227654765023124
    expected_mc_price = -0.6503252107079128
    expected_mc_stddev = 0.4044111449738416
    checks_process_markov_chain(
        cgmy_model,
        forward,
        expected_ioj,
        expected_vol_adj,
        expected_mc_price,
        expected_mc_stddev,
    )
