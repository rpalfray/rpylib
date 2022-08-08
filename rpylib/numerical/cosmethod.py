"""
Implementation of the COS method

This implementation is based on the paper 'A novel pricing method for European options based on Fourier-cosine
series expansions' by Fang and Osterlee.
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray

from ..product.payoff import PayoffType, Vanilla, Forward
from ..product.product import Product


class COSPricer:
    """COS pricer object: this method can price vanilla options for any stochastic process for which the characteristic
    function is known in analytically form.

    .. note:: Extension for Bermudan/American products is not yet implemented.
    """

    def __init__(self, model: "LevyModel", n: int = 10_000, l: int = 10):
        """
        :param model: Lévy stochastic model
        :param n: number of points used in the pricing method
        :param l: cut-off
        """
        from rpylib.model.levymodel.exponentialoflevymodel import ExponentialOfLevyModel

        self.n = n
        self.l = l
        self.model = model
        self.cf = (
            model.log_characteristic_function
            if isinstance(model, ExponentialOfLevyModel)
            else model.characteristic_function
        )
        self._weights = np.ones(n)
        self._weights[0] = 0.5

    def _interval_a_b(self, t: float):
        c1 = self.model.cumulant.cumulant1(t)
        c2 = self.model.cumulant.cumulant2(t)
        c4 = self.model.cumulant.cumulant4(t)
        c6 = 0
        try:
            c6 = self.model.cumulant.cumulant6(t)
        except:
            pass
        delta = self.l * np.sqrt(c2 + np.sqrt(c4 + np.sqrt(c6)))
        return c1 - delta, c1 + delta

    def density_log(self, time: float, u: np.array):
        """
        Log-density function of the Lévy stochastic process
        :param time: time
        :param u: space parameter
        :return: the value of the log-density function in (t,u)
        """
        s = np.exp(u)
        return self.density(time=time, s=s) * s

    def density(self, time: float, s: np.array):
        """
        Density function of the Lévy stochastic process
        :param time: time
        :param s: space parameter
        :return: the value of the density function in (t,s)
        """
        a, b = self._interval_a_b(t=time)
        a += self.model.x0_value()
        b += self.model.x0_value()

        ks = np.arange(self.n)
        cst = ks * np.pi / (b - a)

        fk = 2 / (b - a) * (self.cf(t=time, x=cst) * np.exp(-1j * a * cst)).real
        cosines = np.cos(np.outer(np.log(s) - a, cst))

        return np.sum(cosines * self._weights * fk, axis=1) / s

    def cdf(self, time: float, x: float):
        """
        Cumulative distribution function of the Lévy stochastic process
        i.e. P(S_t < x) where S_t = S_0 exp(X_t) and X_t is a Lévy process
        :param time: time
        :param x: space parameter
        :return: the value of the cumulative function in (t,x)
        """
        return 1 - self.digital(strikes=x, time=time)

    @staticmethod
    def psi(ks: NDArray[np.int], a: float, b: float, c: float, d: float):
        where_to_divide = np.ones(len(ks), dtype=bool)
        where_to_divide[ks == 0] = False
        cst = np.pi / (b - a) * ks
        aux = np.sin(cst * (d - a)) - np.sin(cst * (c - a))
        res = np.divide(aux, cst, where=where_to_divide)
        res[ks == 0] = d - c

        return res

    @staticmethod
    def xi(k: NDArray[np.int], a: float, b: float, c: float, d: float):
        cst = k * np.pi / (b - a)

        def aux1(x):
            return np.cos(cst * (x - a)) * np.exp(x)

        def aux2(x):
            return cst * np.sin(cst * (x - a)) * np.exp(x)

        num = aux1(d) - aux1(c) + aux2(d) - aux2(c)
        den = 1 + cst**2

        return num / den

    @staticmethod
    def u_put(k, a, b):
        return (
            2
            / (b - a)
            * (-COSPricer.xi(k, a, b, a, 0.0) + COSPricer.psi(k, a, b, a, 0.0))
        )

    def _pricing_formula(self, x, time, a, b, vk_coefficients):
        log_spot = self.model.x0_value()
        df = self.model.df(t=time)

        cst = np.arange(self.n) * np.pi / (b - a)
        exp_s = np.exp(1j * np.outer(x - a, cst))
        phi_s = self.cf(t=time, x=cst) * np.exp(-1j * cst * log_spot)
        sum_term = np.sum(
            np.multiply(phi_s * self._weights * vk_coefficients, exp_s), axis=1
        )

        return df * sum_term.real

    def forward(self, strikes, time):
        """
        :param strikes: vector of strikes
        :param time: time maturity
        :return: the price vector of the forward contracts with the given strikes
        """
        from rpylib.model.levymodel.exponentialoflevymodel import ExponentialOfLevyModel

        if isinstance(self.model, ExponentialOfLevyModel):
            fwd = self.model.spot * self.model.mean(time)
        else:
            fwd = self.model.x0_value() + self.model.drift() * time
        df = self.model.df(t=time)
        return df * (fwd - strikes)

    def put(self, strikes, time):
        """
        :param strikes: strike vector
        :param time: time maturity
        :return: the price vector of the put options with the given strikes
        """
        a, b = self._interval_a_b(t=time)
        k = np.arange(self.n)
        u_values = self.u_put(k, a, b)
        spot = self.model.spot
        return strikes * self._pricing_formula(
            np.log(spot / strikes), time, a, b, u_values
        )

    def call(self, strikes, time):
        """
        :param strikes: strike vector
        :param time: time maturity
        :return: the price vector of the call options with the given strikes

            .. note:: the call-put parity formula is used in this case
        """
        return self.forward(strikes=strikes, time=time) + self.put(strikes, time)

    def butterfly(self, strike1, strike2, strike3, time):
        """Price of a butterfly option

        :param strike1: left strike
        :param strike2: middle strike
        :param strike3: right strike
        :param time: time maturity
        :return: the price of the butterfly option
        """
        calls = self.call(np.array([strike1, strike2, strike3]), time)
        return calls[0] - 2 * calls[1] + calls[2]

    def digital(self, strikes, time):
        """Price of a digital option which, at the time maturity, pays 1 if S_T > K else 0

        :param strikes: digital strike
        :param time: digital maturity
        :return: the price vector of the digital options with the given strikes
        """
        a, b = self._interval_a_b(t=time)
        k = np.arange(self.n)
        spot = self.model.spot
        vk = 2 / (b - a) * self.psi(k, a, b, 0.0, b)
        return self._pricing_formula(np.log(spot / strikes), time, a, b, vk)

    def price(self, product: Product) -> Union[float, np.array]:
        """Generic function to price vanilla options with the COS method.

        :param product: vanilla product
        :return: the price of the product given by the COS method

        .. note:: only Forward, Call and Put options are supported for now.
        """
        payoff = product.payoff
        if isinstance(payoff, Forward):
            return self.forward(strikes=payoff.strike, time=product.maturity)
        if isinstance(payoff, Vanilla):
            payoff_type = payoff.payoff_type
            if payoff_type == PayoffType.CALL:
                return self.call(strikes=payoff.strike, time=product.maturity)
            if payoff_type == PayoffType.PUT:
                return self.put(strikes=payoff.strike, time=product.maturity)

        raise NotImplementedError("pricing formula not implemented for the COS method")
