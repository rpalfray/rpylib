"""Testing the time axes"""


import numpy as np

from rpylib.grid.time import TimeGrid


def test_grid_time():
    a = TimeGrid(start=0, end=1, num=11)
    expected_a = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert np.allclose(a, expected_a)
