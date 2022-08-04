"""Multidimensional Lévy process modelled via a Lévy copula

For definitions and properties, the best place to start is the seminal paper:
'Characterization of dependence of multidimensional Lévy processes using Lévy copulas' by Jan Kallsen and Peter Tankov
"""

from collections.abc import Iterator
from functools import partial, lru_cache, reduce
from itertools import product
from typing import Union

import numpy as np
from scipy import optimize

from .levymodel.exponentialoflevymodel import ExponentialOfLevyModel
from .levymodel.levymodel import LevyModel
from .model import Model
from ..distribution.levycopula import LevyCopula
from ..numerical.tools import sign, interval_I
from ..process.process import ProcessRepresentation


def volume(f, a, b) -> float:
    """Volume of the function f over the rectangle [a1, b1]x[a2, b2]x...x[an, bn]

    :param f: function to integrate
    :param a: vector [a1, a2, a3,..., an]
    :param b: vector [b1, b2, b3,..., bn]

    where a <= b, i.e. a1 <= b1, a2 <= b2,...

    :return: the integral over the rectangle :math:`a x b`
    """

    res = 0
    n = len(a)
    for p in product([0, 1], repeat=n):
        u = (ai if pi == 0 else bi for pi, ai, bi in zip(p, a, b))
        n_power = n - sum(p)
        factor = -1 if n_power % 2 else 1
        res += factor*f(u)

    return res


def margin(f, indices: list[int], dimension: int):
    """I-margin of the f function

    :param f: function f
    :param indices: indices of I
    :param dimension: dimension of the function
    :return: I-margin of f
    """
    if indices is None:
        return f

    c_indices = list(set(range(dimension)) - set(indices))
    nb = len(c_indices)

    def i_margin(u) -> float:
        u_array = np.zeros(shape=dimension)
        u_array[list(indices)] = list(u)
        res = 0
        for p in product([-np.inf, np.inf], repeat=nb):
            u_array[c_indices] = p
            this_sign = reduce(lambda x, y: sign(x)*sign(y), p, 1.0)
            res += f(u_array)*this_sign

        return res

    return i_margin


class LevyCopulaModel(Model):
    """Multidimensional Lévy process modelled via a Lévy copula
    """
    def __init__(self, models: [LevyModel], copula: LevyCopula):
        """
        :param models: processes characterising the margins of the copula
        :param copula: copula function characterising the dependence of the margins
        """
        super().__init__()
        # for simplicity, we assume that the underlyings models are either all ExponentialOfLévyModels or all LévyModel
        # (i.e. in the sense that none of them are ExponentialOfLévyModels)
        all_exp_of_levy_model = all(isinstance(model, ExponentialOfLevyModel) for model in models)
        none_exp_of_levy_model = all(not isinstance(model, ExponentialOfLevyModel) for model in models)
        if not (all_exp_of_levy_model or none_exp_of_levy_model):
            raise ValueError('Expected all models or none of them to be ExponentialOfLevyModel')

        self.process_representation = ProcessRepresentation.Identity
        if all_exp_of_levy_model:
            self.process_representation = ProcessRepresentation.Log

        self.models = models
        self.copula = copula

        self._dimension = len(models)
        self.x0s = np.array([[model.x0_value() for model in models]]).T
        self._marginal_levy_measure = [model.levy_triplet.nu for model in self.models]
        self._full_indices = list(range(self._dimension))

        self.mass = self._mass_nd
        if len(models) == 2:
            self.mass = self._mass_2d  # hard-coded version for 2d version
        elif len(models) == 3:
            self.mass = self._mass_3d  # hard-coded version for 3d version

    def __repr__(self):
        return 'LevyCopulaModel(models=models, copula=copula)'.format(model=self.models, copula=self.copula)

    def dimension(self) -> int:
        return self._dimension

    def truncate_levy_measure(self, truncations) -> None:
        for model, truncation in zip(self.models, truncations):
            model.truncate_levy_measure(truncations=truncation)

    def x0_value(self):
        return self.x0s

    def df(self, t: float) -> float:
        return self.models[0].df(t)

    def jump_of_finite_activity(self) -> bool:
        return self.blumenthal_getoor_index() == 0  # FIXME: not mathematically true

    def jump_of_finite_variation(self) -> bool:
        return self.blumenthal_getoor_index() <= 1

    def finite_first_moment(self):
        return all(model.finite_first_moment() for model in self.models)

    def levy_exponent(self, x: Union[complex, list[complex]]) -> complex:
        raise NotImplementedError('not implemented yet for Levy copula')

    def blumenthal_getoor_index(self) -> float:
        return max(model.blumenthal_getoor_index() for model in self.models)

    def characteristic_function(self, t: float, x: complex) -> complex:
        raise NotImplementedError('not implemented yet for Levy copula')

    def cdf(self, t: float, x: 'np.array'):
        raise NotImplementedError('not implemented yet for Levy copula')

    def _mass_nd(self, a, b, indices: list[int] = None):
        """Integration of the I-margin of the Lévy measure over [a1, b1]x[a2, b2]x...x[an, bn]
        indices is the set I, if None, it defaults to the Lévy measure

        :param a: vector [a1, a2, a3,..., an]
        :param b: vector [b1, b2, b3,..., bn]
        :param indices: set of integers which characterises the I-margin

        where a <= b, i.e. a1 <= b1, a2 <= b2,...

        :return: the integral over the rectangle (a, b]
        """
        if indices is None:
            indices = self._full_indices

        j = next((i for i, ai, bi in zip(indices, a, b) if ai < 0 < bi), None)
        if j is not None:
            k = indices.index(j)
            a_1, b_1 = list(a), list(b)
            a_2, b_2 = list(a), list(b)
            a_1[k], b_1[k] = b[k], np.inf
            a_2[k], b_2[k] = -np.inf, a[k]

            indices_j = list(indices)
            a_j, b_j = list(a), list(b)
            indices_j.pop(k)
            a_j.pop(k)
            b_j.pop(k)
    
            j_mass = self._mass_nd(a_j, b_j, indices_j)
            m1 = self._mass_nd(a_1, b_1, indices)
            m2 = self._mass_nd(a_2, b_2, indices)

            return j_mass - m1 - m2
        else:
            i_tail_integrals = partial(self.margin_tail_integral, indices)
            eps = -1 if len(a) % 2 else 1
            res = volume(i_tail_integrals, a, b)
            return eps*res

    def _mass_1d(self, a, b, index):
        u = partial(self.marginal_tail_integral, index)
        return u(a) - u(b)

    def _mass_2d(self, a, b, indices: list[int] = None):
        if indices is not None and len(indices) == 1:
            return self._mass_1d(a[0], b[0], indices[0])

        if indices is None:
            indices = [0, 1]

        i1, i2 = indices
        a1, a2 = a
        b1, b2 = b

        aux = 0
        if a1 < 0 < b1:
            aux = self._mass_1d(a2, b2, i2)

        if a2 < 0 < b2:
            aux = self._mass_1d(a1, b1, i1)

        u = partial(self.margin_tail_integral, indices)

        return u(a) + u(b) - u((a1, b2)) - u((b1, a2)) + aux

    def _mass_3d(self, a, b, indices: list[int] = None):
        if indices is not None and len(indices) < 3:
            return self._mass_2d(a, b, indices)

        if indices is None:
            indices = [0, 1, 2]

        i1, i2, i3 = indices
        a1, a2, a3 = a
        b1, b2, b3 = b

        aux = 0
        if a1 < 0 < b1:
            aux = self._mass_2d((a2, a3), (b2, b3), indices=[i2, i3])
            if a2 < 0 < b2:
                u13 = partial(self.margin_tail_integral, [i1, i3])
                aux += u13((a1, a3)) - u13((a1, b3)) - u13((b1, a3)) + u13((b1, b3))
            elif a3 < 0 < b3:
                u12 = partial(self.margin_tail_integral, [i1, i2])
                aux += u12((a1, a2)) - u12((a1, b2)) - u12((b1, a2)) + u12((b1, b2))
        elif a2 < 0 < b2:
            aux = self._mass_2d((a1, a3), (b1, b3), indices=[i1, i3])
            if a1 < 0 < b1:
                u23 = partial(self.margin_tail_integral, [i2, i3])
                aux += u23((a2, a3)) - u23((a2, b3)) - u23((b2, a3)) + u23((b2, b3))
            elif a3 < 0 < b3:
                u12 = partial(self.margin_tail_integral, [i1, i2])
                aux += u12((a1, a2)) - u12((a1, b2)) - u12((b1, a2)) + u12((b1, b2))
        elif a3 < 0 < b3:
            aux = self._mass_2d((a1, a2), (b1, b2), indices=[i1, i2])
            if a1 < 0 < b1:
                u23 = partial(self.margin_tail_integral, [i2, i3])
                aux += u23((a2, a3)) - u23((a2, b3)) - u23((b2, a3)) + u23((b2, b3))
            elif a2 < 0 < b2:
                u13 = partial(self.margin_tail_integral, [i1, i3])
                aux += u13((a1, a3)) - u13((a1, b3)) - u13((b1, a3)) + u13((b1, b3))

        u = partial(self.margin_tail_integral, indices)

        vol = u(a) - u(b) \
              - u((a1, a2, b3)) - u((a1, b2, a3)) + u((a1, b2, b3)) \
              - u((b1, a2, a3)) + u((b1, a2, b3)) + u((b1, b2, a3))

        return aux + vol

    def drift(self, t: float = 0, x: np.array = 0) -> np.array:
        return np.array([[model.drift() for model in self.models]]).T

    def diffusion_coefficient(self) -> float:
        # this should return the diffusion matrix
        raise NotImplementedError('not implemented yet for Levy copula')

    def plot_density(self, t: float, show: bool = False) -> None:
        raise NotImplementedError('not implemented yet for Levy copula')

    def plot_cdf(self, t: float, data: np.array, log_normalisation: bool = True, show: bool = False,
                 title='') -> None:
        raise NotImplementedError('not implemented yet for Levy copula')

    @lru_cache(maxsize=2**10)
    def marginal_tail_integral(self, i: int, x: float) -> float:
        """Tail integral of the i-th marginal"""
        return sign(x)*self._marginal_levy_measure[i].integrate(*interval_I(x))

    def margin_tail_integral(self, indices: list[int], x: Iterator[int]):
        if indices == self._full_indices:
            return self.tail_integrals(x)

        if len(indices) == 1:
            i0, x0 = indices[0], next(x)
            return self.marginal_tail_integral(i0, x0)

        i_copula = margin(self.copula, indices, self._dimension)  # args: f, indices, dimension - not passed as kwarg
        # for (slight) optimisation purpose
        return i_copula(np.array([self.marginal_tail_integral(i, xi) for i, xi in zip(indices, x)]))

    def tail_integrals(self, x) -> float:
        """Calculate the tail integral of the I-margin of the Lévy copula

        :param x: vector of values
        :return: :math:`F(P^i_1(x_1), P^i_2(x_2),...P^i_n(x_n))` for :math:`i \\in indices` where
                 :math:`P^i_j(x) = sgn(x)*\\nu^j(I(x))` with:

                     * :math:`I(x) = (x, \\inf)` if :math:`x\\geq 0`
                     * and :math:`I(x) = (-\\inf, x]` if :math:`x<0`
        """
        return self.copula(np.array([self.marginal_tail_integral(i, xi) for i, xi in enumerate(x)]))

    def inverse_tail_integral(self, i, x):

        def fun_root(u: float) -> float:
            return self.marginal_tail_integral(i=i, x=u) - x

        if x > 0:
            a, b = 1e-20, 500.0
            if fun_root(a) < 0:
                return a
        else:
            a, b = -500.0, -1e-20
            if fun_root(b) > 0:
                return b

        solution = optimize.toms748(f=fun_root, a=a, b=b, xtol=1e-14)
        return solution
