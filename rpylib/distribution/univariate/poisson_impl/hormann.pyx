""""
Poisson generator as in Hormann W (1993) "The transformed rejection method for generating Poisson random variables"
"""

# cython: infer_types = True

import numpy as np
from libc.stdlib cimport rand, RAND_MAX
from scipy.special import gammaln

cdef double RAN_MAX_DOUBLE
RAN_MAX2_DOUBLE = <double>(RAND_MAX-1)


cdef class Hormann:

    cdef public double lam
    cdef double _loglam, _beta, _alpha, _k

    def __init__(self, lam: float):
        self.lam = lam
        self._loglam = np.log(lam)
        c =   0.767 - 3.36/lam
        self._beta = np.pi/np.sqrt(3*lam)
        self._alpha = self._beta*lam
        self._k = np.log(c) - lam - np.log(self._beta)

    def cost(self):
        raise NotImplementedError

    def sample(self, int size=1):
        alpha, beta, k, loglam  = self._alpha, self._beta, self._k, self._loglam

        cdef np.ndarray[np.npy_double, ndim=1] res = np.empty(shape=size, dtype=np.double)
        cdef int cost
        cdef int i
        cost = 0

        if size:
            for i in range(size):
                res[i] = draw_one(alpha, beta, k, loglam)
        else:
            res[0] = draw_one(alpha, beta, k, loglam)

        return res


cdef double random_sample(): # return uniform in [0,1)
    return rand()/RAN_MAX2_DOUBLE


cdef double draw_one(double alpha, double beta, double k, double loglam):
    cdef int cost = 0
    cdef double x, y, lhs, rhs
    cdef float u, v
    cdef int n

    while True:
        u = random_sample()
        cost += 1
        x = (alpha - np.log(1-u)/u)/beta
        n = int(np.floor(x + 0.5))
        if n < 0:
            continue
        v = random_sample()
        cost += 1
        y = alpha - beta*x
        lhs = y + np.log(v/(1 + np.exp(y))**2)
        rhs = k + n*loglam - gammaln(n+1)
        if lhs <= rhs:
            return n
