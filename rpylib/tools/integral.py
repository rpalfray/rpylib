"""A few useful integrals
"""

from math import factorial

import numpy as np
import scipy.special


def _helper_sum_fact_xk(n, x):
    """Compute sum over k of n!/k! * |x|^k for k=0...n"""
    n_fact = factorial(n)
    k_factorial = scipy.special.factorial(np.arange(n+1))
    x_power = np.power(abs(x), np.arange(n+1))
    res = n_fact*np.dot(x_power, k_factorial)

    return res


def integral_xn_exp_minus_x(n: int, a: float, b: float, alpha: float):
    """Integral of x^n exp(-alpha*|x|) over [a, b] where alpha>0"""
    if alpha <= 0:
        raise ValueError('Expected alpha > 0')

    if a < 0 < b:
        return integral_xn_exp_minus_x(n=n, a=a, b=0., alpha=alpha) \
               + integral_xn_exp_minus_x(n=n, a=0., b=b, alpha=alpha)

    aux = alpha**(n+1)

    def helper(u):
        return _helper_sum_fact_xk(n, u*alpha)*np.exp(-abs(u)*alpha)/aux

    if a == -np.inf:
        return -helper(b)

    if b == np.inf:
        return helper(a)

    return helper(a) - helper(b)
