"""Generic helper functions"""


from functools import singledispatch

import numpy as np


@singledispatch
def sign(x):
    eps = 1.0
    for xi in x:
        eps *= -1.0 if xi < 0 else 1.0
    return eps


@sign.register
def _(x: float):
    if x < 0:
        return -1.0

    return 1.0


def interval_I(x: float) -> tuple[float, float]:
    """
    :return: left or right interval depending on the sign of x
    """
    return (-np.inf, x) if x < 0 else (x, np.inf)
