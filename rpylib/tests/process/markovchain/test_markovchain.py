"""Testing the Markov-Chain process implementation
"""

import numpy as np

from rpylib.distribution.sampling import SamplingMethod
from rpylib.montecarlo.path import StochasticJumpPath
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
    expected_log_diffusion_path,
    expected_log_jump_path,
):
    grid = spatial_grid(model=model)
    mcp = MarkovChainProcess(model=model, grid=grid, method=SamplingMethod.ALIAS)
    ioj = mcp.intensity_of_jumps
    vol_adj = vol_adjustment(model=model, h=grid.h)

    mcp.initialisation(product=product)
    mc_paths = 1
    mcp.pre_computation(mc_paths=mc_paths, product=product)

    # FIXME: force the simulation to be deterministic
    # simulation: StochasticJumpPath = None
    # for i in range(mc_paths):
    #     simulation = mcp.simulate_one_path()

    assert np.isclose(ioj, expected_ioj)
    assert np.isclose(vol_adj, expected_vol_adj)
    # assert(np.allclose(simulation.diffusion_path, expected_log_diffusion_path))
    # assert(np.allclose(simulation.jump_path, expected_log_jump_path))


def test_hem_process_markov_chain_testing_value(hem_model, forward):
    expected_ioj = 4.28495221390975
    expected_vol_adj = 0.0
    expected_log_diffusion_path = [0.0, -0.00734127]
    expected_log_jump_path = [0.0, 0.0]
    checks_process_markov_chain(
        hem_model,
        forward,
        expected_ioj,
        expected_vol_adj,
        expected_log_diffusion_path,
        expected_log_jump_path,
    )


def test_cgmy_process_markov_chain_testing_value(cgmy_model, forward):
    expected_ioj = 26.166020822055167
    expected_vol_adj = 0.030227654765023124
    expected_log_diffusion_path = [0.0, 0.00152005]
    expected_log_jump_path = [0.0, -0.02]
    checks_process_markov_chain(
        cgmy_model,
        forward,
        expected_ioj,
        expected_vol_adj,
        expected_log_diffusion_path,
        expected_log_jump_path,
    )
