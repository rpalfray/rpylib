"""The LévyDrivenSDE class models a Lévy-driven SDEs where the driver is a pure jump process,
that is the represented process X is solution of the SDE: dX = h(X) dY, X(0) = X_0 where Y is a pure jump process.

Functions h are objects of the class SDEFunction, assumed to be "regular" function (at least be Lipschitz).
"""

from numbers import Number
from typing import Union

import numpy as np

from rpylib.model.levycopulamodel import LevyCopulaModel
from rpylib.model.levymodel.levymodel import LevyModel
from rpylib.model.model import Model
from rpylib.process.process import ProcessRepresentation


class SDEFunction:
    """Function h corresponding to the SDE dX = h(X) dYs
        h is function from R^m to R^(mxd) where m is the dimension of X and d the dimension of the Lévy driver Y
    """
    def __init__(self, m: int, d: int):
        """
        :param m: dimension of the modelled underlying
        :param d: dimension of the SDE driver
        """
        self.shape = (m, d)

    def __call__(self, t: float, x: np.array) -> np.array:
        pass


class Constant(SDEFunction):
    """Constant function h"""
    def __init__(self, m: int = 1, d: int = 1, constant: float = 1):
        super().__init__(m=m, d=d)
        self.constant_matrix = np.full(shape=(m, d), fill_value=constant)

    def __call__(self, t: float, x: np.array) -> np.array:
        return self.constant_matrix


class DiagX(SDEFunction):
    """h(X) returns diag(x1, x2,..., xn).
    This only works when m=d i.e. when the driver has the same dimension as the underlying.
    """

    def __init__(self, dimension: int = 1):
        """
        :param dimension: dimension of the driver (=number of modelled underlyings)
        """
        super().__init__(m=dimension, d=dimension)

    def __call__(self, t: float, x: np.array) -> np.array:
        return np.diag(x)


class LiborSDEFunction(SDEFunction):
    """The function h is such that h(x) = sigma where sigma is an array of size mxd with m the number of underlying
    Libor rates and d the model dimension"""
    def __init__(self, sigma: np.array, tenors: np.array):
        """
        :param sigma: matrix sigma
        :param tenors: tenors of the underlying Libor rates
        """
        m, d = sigma.shape
        super().__init__(m=m, d=d)
        self._sigma = sigma
        self.tenors = tenors

    def sigma(self, t: float):
        """The sigma coefficient corresponding to the Libor with tenor T is zero for t >= T (the Libor rate fixes at T)
        :param t: time t
        """
        if self.tenors[0] > t:
            return self._sigma
        else:
            res = self._sigma.copy()
            res[np.argwhere(self.tenors[:-1] <= t)] = 0
            return res

    def __call__(self, t: float, x: np.array) -> np.array:
        return self.sigma(t)*x


class ForwardMarketSDEFunction(SDEFunction):
    """The function h is such that h(x) = sigma where sigma is an array of size mxd with m the number of underlying
    OIS rates and d the model dimension"""
    def __init__(self, sigma: np.array, tenors: np.array):
        """
        :param sigma: matrix sigma
        :param tenors: tenors of the underlying Libor rates
        """
        m, d = sigma.shape
        super().__init__(m=m, d=d)
        self._sigma = sigma
        self.tenors = tenors

    def sigma(self, t: float):
        """The sigma coefficient corresponding to the OIS term rate for the period [Ti, Ti+1]. It is 0 for t >= Ti+1 and
        decreasing between Ti and Ti+1. The convention is to take sigma linearly decreasing between Ti and Ti+1.
        :param t: time t
        """
        if self.tenors[0] > t:
            return self._sigma
        else:
            res = self._sigma.copy()
            g = np.minimum(1, np.maximum(0, self.tenors - t)/(self.tenors[1:] - self.tenors[:-1]))
            res = res*np.diag(g)
            return res

    def __call__(self, t: float, x: np.array) -> np.array:
        return self.sigma(t)*x


LevyDriver = Union[LevyModel, LevyCopulaModel]


class LevyDrivenSDEModel(Model):
    """Representation of the process X which is solution of: dX = h(X) dY, X(0) = X_0 where Y is a pure jump Lévy
    process
    """
    process_representation = ProcessRepresentation.IDENDITY

    def __init__(self, driver: LevyDriver, x0: Union[float, np.array] = 0., a: SDEFunction = None):
        """
        :param driver: SDE driver
        :param x0: initial value of X
        :param a: SDE Function h (FIXME change of notation might be a bit confusing)
        """
        super().__init__()
        self._m = 1 if isinstance(x0, Number) else x0.size
        self._d = driver.dimension()

        self.x0 = np.atleast_1d(x0)
        self.a = a or Constant(m=self._m, d=self._d)
        self.driver = driver

        # check consistency with the dimension
        m, d = self.a.shape
        if d != self._d or m != self._m:
            raise ValueError('Expected \'a\' to be a function from R^m to R^(mxd) where m is the dimension of x0(={}) '
                             'and d is the dimension of the driver(={}), found m={} and d={}'
                             .format(self._m, self._d, m, d))

    def truncate_levy_measure(self, truncations) -> None:
        self.driver.truncate_levy_measure(truncations=truncations)

    def blumenthal_getoor_index(self) -> float:
        return self.driver.blumenthal_getoor_index()

    def dimension(self) -> int:
        return self._m

    def dimension_model(self) -> int:
        return self._d

    def x0_value(self):
        return self.x0

    def drift(self, t: float = 0, x: np.array = 0) -> np.array:
        return np.zeros_like(x)

    def df(self, t: float) -> float:
        return 1.0
