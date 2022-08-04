"""Lévy Copula

For general definitions and properties, see the seminal paper
'Characterization of dependence of multidimensional Lévy processes using Lévy copulas' by Kallsen and Tankov
"""

import abc

import numpy as np
from scipy import optimize

from rpylib.numerical.tools import sign
from rpylib.tools.parameter import strictly_positive


class LevyCopula:

    @abc.abstractmethod
    def __call__(self, us: np.array) -> float:
        """copula function F"""

    def conditional_distribution(self, eps: float, x: np.array) -> np.array:
        """The conditional distribution :math:`F_{\\mathcal{E}}(x)` as defined in
        'Lévy copulas: review of recent results' by Peter Tankov
        """
        raise NotImplementedError('The conditional_distribution function is not implemented for this Levy copula')

    def inverse_conditional_distribution(self, eps: np.array, x: np.array) -> np.array:
        def inverse_root(y):
            return x - self.conditional_distribution(eps, y)
        solution = optimize.root_scalar(inverse_root, method='newton')
        return solution.root

    def x_first_derivative(self, u: np.array) -> float:
        """Derivative of the Lévy copula with regard to :math:`u_i, u_j,\\dots` times the product :math:`u_i*u_j*\\dots`

        :param: :math:`u` -> variables :math:`u_i, u_j,\\dots`
        """
        raise NotImplementedError('Copula function: first derivative not yet implemented')


class ClaytonCopula(LevyCopula):
    """The Clayton copula function is parametrised by theta and eta. See formula (7) in
    'Lévy copulas: review of recent results' by Peter Tankov
    """
    theta = strictly_positive('theta')

    def __init__(self, theta: float, eta: float):
        if not 0.0 <= eta <= 1.0:
            raise ValueError('expected eta in [0,1]')
        self.theta = theta
        self.eta = eta

    def __repr__(self):
        return 'ClaytonCopula(theta={:2f}, eta={:2f})'.format(self.theta, self.eta)

    def __call__(self, us: np.array) -> float:
        if 0 in us:
            return 0.0

        sign_us = np.sign(us)
        sign_prod = 1
        sum_elmts = 0
        for elmt, sign_u in zip(us, sign_us):
            sum_elmts += abs(elmt)**(-self.theta)
            sign_prod *= sign_u

        # note that it seems to be slightly faster than:
        # sum_elmts = sum(abs(elmt)**(-self.theta) for elmt in us)
        # or
        # sum_elmts = np.sum(np.absolute(us)**(-self.theta))

        factor = self.eta if sign_prod >= 0 else -(1.0 - self.eta)
        return 2**(2 - us.size)*(sum_elmts**(-1.0/self.theta))*factor

    def conditional_distribution(self, eps: float, x: np.array) -> np.array:
        if x.size == 1:
            return self._condition_distribution_2d(eps, x)
        else:
            raise NotImplementedError('not implemented yet for Levy copula for d>2')

    def _condition_distribution_2d(self, eps: float, x: np.array) -> np.array:
        eta, theta = self.eta, self.theta

        if eps >= 0:
            aux = 1 if x[0] < 0 else 0
            res = 1 - eta + np.power(1 + np.power(abs(eps/x[0]), theta), -1 - 1/theta)*(eta - aux)
        else:
            aux = 1 if x[0] >= 0 else 0
            res = eta + np.power(1 + np.power(abs(eps/x[0]), theta), -1 - 1/theta)*(aux - eta)

        return np.array([res])

    def inverse_conditional_distribution(self, eps: np.array, x: np.array) -> np.array:
        if len(x.shape) == 1:
            return self._inverse_conditional_distribution_2d(eps, x)
        else:
            raise NotImplementedError('not implemented for d>2')

    def _inverse_conditional_distribution_2d(self, eps: np.array, x: np.array) -> np.array:
        eta, theta = self.eta, self.theta

        def fun_b(e, u):
            res_b = np.where(e >= 0, np.sign(u - 1 + eta), np.sign(u - eta))
            return res_b

        def fun_c(e, u):
            res_c = np.where(e >= 0,
                             np.where(u >= 1 - eta, (u - 1 + eta)/eta, (1 - eta - u)/(1 - eta)),
                             np.where(u >= eta, (u - eta)/(1 - eta), (eta - u)/eta))
            return res_c

        res = fun_b(eps, x)*np.abs(eps)*np.power(np.power(fun_c(eps, x), -theta/(theta + 1)) - 1, -1/theta)
        return res

    def x_first_derivative(self, u: np.array) -> float:
        if np.any(u == 0):
            return 0

        dim = u.size
        theta = self.theta
        u_prod = np.prod(u)
        theta_prod = np.prod(1 + np.arange(dim)*theta)
        factor = self.eta if u_prod >= 0 else -(1.0 - self.eta)

        res = 2**(2 - dim) * theta_prod * factor
        term1 = abs(u_prod)**(-theta-1)
        term2 = np.sum(np.power(np.abs(u), -theta))**(-1/theta-dim)
        aux = term1*term2

        res *= aux
        return res


class IndependentComponentsCopula(LevyCopula):
    """In this case, the margins are independent, see formula (4.2) in
    'Characterization of dependence of multidimensional Lévy processes using Lévy copulas' by Kallsen and Tankov
    """

    def __call__(self, us: np.array) -> float:
        kronecker_symbols = np.zeros_like(us)
        kronecker_symbols[us == np.inf] = 1.0

        res = 0
        for k, u in enumerate(us):
            if not np.isinf(u):
                product = np.prod(kronecker_symbols[:k])*np.prod(kronecker_symbols[k+1:])
                res += u*product

        return res

    def __repr__(self):
        return 'IndependentComponentsCopula()'


class DependentComponentsCopula(LevyCopula):
    """In this case, the margins are completely dependent, see formula (4.3) in
        'Characterization of dependence of multidimensional Lévy processes using Lévy copulas' by Kallsen and Tankov
    """

    def __call__(self, us: np.array) -> float:
        if np.all(us > 0):
            res = np.amin(us)
        elif np.all(us < 0):
            eps = -1 if us.size % 2 else +1
            res = -np.amax(us)*eps
        else:
            res = 0

        return res

    def conditional_distribution(self, eps: float, x: np.array) -> np.array:
        return np.count_nonzero(x == np.inf)

    def inverse_conditional_distribution(self, eps: np.array, x: np.array) -> np.array:
        raise ValueError("not invertible")

    def __repr__(self):
        return 'DependentComponentsCopula()'


class FrankLevyCopula(LevyCopula):
    """The Frank-Lévy copula is parametrised by eta, see the formula (32) (2d case)
    in 'A Structural Jump Threshold Framework for Credit Risk' by Garreau and Kercheval
    """

    eta = strictly_positive('eta')

    def __init__(self, eta: float):
        self.eta = eta

    def __repr__(self):
        return 'FrankLevyCopula(eta={:2f})'.format(self.eta)

    def __call__(self, us: np.array) -> float:
        if 0 in us:
            return 0.0

        aux = np.log(1 - np.prod(1 - np.exp(-self.eta*np.abs(us))))
        eps = 1
        for u in us:
            eps *= sign(u)

        res = -eps*aux/self.eta
        return res
