"""Functions related to number theory


functions inv_guess_a and upper_bound_a_n are inspired
from http://www.mac-guyver.com/switham/2020/03/HyperbolicPairing/hyperbolic_pairing_v0.5.pdf
"""

from functools import cache
from math import floor, sqrt, log

import numpy as np
import scipy.optimize
from sympy import EulerGamma

euler_gamma = float(EulerGamma)


@cache
def a_n(n):
    """See https://oeis.org/A006218 for more details

    :return: the sequence of the sum of the divisor of k for k in [1,n]
    """
    sqrt_x = floor(sqrt(n))
    res = 2*sum(n // k for k in range(1, sqrt_x+1)) - sqrt_x**2

    return res


def inv_guess_a(c):
    """Helper function"""
    if c < 2:
        return floor(c)

    log_c = log(c)

    def f_fp_fp2(log_n):
        aux = log_n + 2*euler_gamma - 1
        f = log_n - log_c + log(aux)
        fp = 1 + 1/aux
        fp2 = -1/aux**2
        return f, fp, fp2

    sol = scipy.optimize.root_scalar(f=f_fp_fp2, x0=log_c, fprime2=True, method='halley')
    res = sol.root
    return int(np.exp(res))


def upper_bound_a_n(z):
    """Find the "inverse" of a_n defined as n such that a_n(n-1) <= z

    We know that a_n is an increasing function
    """
    if z == 0:
        return 1

    delta_c = 3 * z**(1/4)
    n_low_bound = inv_guess_a(max(0, z - delta_c))
    n_guess = inv_guess_a(z)
    n_high_bound = inv_guess_a(z + delta_c)

    a_guess = a_n(n_guess)
    if a_guess == z:
        return n_guess + 1

    if a_guess > z:
        start, end = n_low_bound, n_guess
    else:
        start, end = n_guess, n_high_bound

    while (increment := end - start) > 1:
        middle = start + increment // 2
        a_middle = a_n(middle)
        if a_middle > z:
            end = middle
        else:
            start = middle

    res = start + 1
    return res
