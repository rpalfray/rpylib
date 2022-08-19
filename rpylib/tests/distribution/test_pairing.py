"""Testing the pairing functions and their inverse"""


import pytest

from rpylib.distribution.pairing import (
    Cantor,
    RosenbergStrong,
    Szudzik,
    HyperbolicPairing,
    PairingToZd,
    PairingToZ1d,
)
from rpylib.distribution.pairing import mapping_to_z, projection_to_z


def test_mapping_to_z():
    """mapping 0, 1, -1, 2, -2, 3, -3,... to 0, 1, 2, 3, 4, 5, 6,..."""
    inputs = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10]
    expected = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    outputs = [mapping_to_z(i) for i in inputs]

    assert expected == outputs


def test_inverse_mapping_to_z():
    """mapping 0, 1, 2, 3, 4, 5, 6,... to 0, 1, -1, 2, -2, 3, -3,..."""
    inputs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    expected = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10]
    outputs = [projection_to_z(i) for i in inputs]

    assert expected == outputs


@pytest.mark.parametrize(
    "pairing", [Cantor, RosenbergStrong, Szudzik, HyperbolicPairing]
)
def test_pairing_to_n(pairing):
    pairing_instance = pairing()
    pairs = [pairing_instance.projection(i) for i in range(100)]
    paired = [pairing_instance.pairing(p) for p in pairs]
    expected = [i for i, _ in enumerate(paired)]

    assert paired == expected


@pytest.mark.parametrize(
    "pairing", [Cantor, RosenbergStrong, Szudzik, HyperbolicPairing]
)
def test_pairing_to_z(pairing):
    pairing_instance = PairingToZd(pairing())
    pairs = (pairing_instance.projection(i) for i in range(100))
    paired = [pairing_instance.pairing(p) for p in pairs]
    expected = [i for i, _ in enumerate(paired)]

    assert paired == expected


def test_mapping_1d():
    grid_size = 7 + 13 + 1
    pairing_a1 = PairingToZ1d((-7, 13), omit_zero=True)
    pairing_a2 = PairingToZ1d((-7, 13), omit_zero=False)
    inputs = list(range(grid_size - 1))
    outputs_a1 = [pairing_a1.project(k) for k in inputs]
    outputs_a2 = [pairing_a2.project(k) for k in inputs]
    expected_a1 = [
        1,
        -1,
        2,
        -2,
        3,
        -3,
        4,
        -4,
        5,
        -5,
        6,
        -6,
        7,
        -7,
        8,
        9,
        10,
        11,
        12,
        13,
    ]
    expected_a2 = [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, 9, 10, 11, 12]
    test_inputs_a1 = [pairing_a1.pair(k) for k in expected_a1]
    test_inputs_a2 = [pairing_a2.pair(k) for k in expected_a2]

    pairing_b1 = PairingToZ1d((-13, 7), omit_zero=True)
    pairing_b2 = PairingToZ1d((-13, 7), omit_zero=False)
    outputs_b1 = [pairing_b1.project(k) for k in range(grid_size - 1)]
    outputs_b2 = [pairing_b2.project(k) for k in range(grid_size - 1)]
    expected_b1 = [
        1,
        -1,
        2,
        -2,
        3,
        -3,
        4,
        -4,
        5,
        -5,
        6,
        -6,
        7,
        -7,
        -8,
        -9,
        -10,
        -11,
        -12,
        -13,
    ]
    expected_b2 = [
        0,
        1,
        -1,
        2,
        -2,
        3,
        -3,
        4,
        -4,
        5,
        -5,
        6,
        -6,
        7,
        -7,
        -8,
        -9,
        -10,
        -11,
        -12,
    ]
    test_inputs_b1 = [pairing_b1.pair(k) for k in expected_b1]
    test_inputs_b2 = [pairing_b2.pair(k) for k in expected_b2]

    assert outputs_a1 == expected_a1 and test_inputs_a1 == inputs
    assert outputs_a2 == expected_a2 and test_inputs_a2 == inputs
    assert outputs_b1 == expected_b1 and test_inputs_b1 == inputs
    assert outputs_b2 == expected_b2 and test_inputs_b2 == inputs


@pytest.mark.parametrize("pairing", [Szudzik, RosenbergStrong, HyperbolicPairing])
def test_mapping_3d(pairing):
    pairing1 = PairingToZd(pairing(), dimension=3, omit_zero=True)
    inputs = list(range(100))
    pairs1 = [pairing1.project(i) for i in inputs]
    paired1 = [pairing1.pair(p) for p in pairs1]

    pairing2 = PairingToZd(pairing(), dimension=3, omit_zero=False)
    pairs2 = [pairing2.project(i) for i in inputs]
    paired2 = [pairing2.pair(p) for p in pairs2]

    assert paired2 == inputs
    assert paired1 == inputs
