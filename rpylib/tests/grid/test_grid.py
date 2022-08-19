"""Testing the axes class"""


import pytest

from rpylib.grid.grid import Uniform1DGrid


def test_grid_uniform_start_end():
    with pytest.raises(ValueError):
        Uniform1DGrid(start=1.0, end=0.0, num=10)


def test_grid_uniform_negative_step():
    with pytest.raises(ValueError):
        Uniform1DGrid(start=0, end=10, num=-1)
