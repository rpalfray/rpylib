"""Fast-Fourier Transform pricer

This is the implementation of call/put pricing from 'Option Valuation Using the Fast Fourier Transform'
by Carr and Madan
"""

from typing import Union

import numpy as np


class FFTPricer:
    """Fast Fourier Transform pricer
    """

    def __init__(self, model: 'LevyModel'):
        """
        :param model: Lévy model
        """
        self.cf = model.log_characteristic_function
        self.r = model.r
        self.model = model

        self.alpha = 1.5
        self.eta = 0.25
        self.N = 2**18

        self.l = 2*np.pi/(self.N*self.eta)
        self.b = -model.x0_value() + np.pi/self.eta

    def _sufficient_condition(self, t: float) -> None:
        moment = self.cf(t=t, x=-1j*(1 + self.alpha))
        if moment.real > 1e10:
            raise ValueError('sufficient condition on alpha probably not met in the fft pricer')

    def _psi(self, t: float, v: np.array) -> float:
        alpha = self.alpha
        res = np.exp(-self.r*t) * self.cf(t=t, x=v - (1+alpha)*1j) / (alpha*(1+alpha) - v*v + 1j*(2*alpha+1)*v)
        return res

    def _call_prices(self, maturity: float):
        self._sufficient_condition(maturity)
        n = self.N

        vs = np.arange(n)*self.eta   # v_i
        ds = np.zeros(n)             # kronecker symbol
        ds[0] = 1

        w_s = self.eta/3*(3.0 + (-1)**(np.arange(n)+1) - ds)  # weights for the Simpson's rule
        psi_s = self._psi(t=maturity, v=vs)
        a_s = np.exp(1j*self.b*vs) * psi_s * w_s

        fft_prices = np.fft.fft(a_s)

        ks = -self.b + np.arange(n)*self.l  # log strikes
        res = np.exp(-self.alpha*ks)/np.pi * fft_prices.real
        return ks, res

    def call(self, strike: Union[float, list[float]], maturity: float) -> float:
        """Call price

        :param strike: call strike
        :param maturity: call maturity
        :return: the value of the call option
        """
        res_strikes, res_prices = self._call_prices(maturity)
        prices = np.interp(np.log(strike), res_strikes, res_prices)
        return prices

    def put(self, strike: Union[float, list[float]], maturity: float) -> float:
        """Put price calculated from the call/put parity formula

        :param strike: put strike
        :param maturity: put maturity
        :return: the value of the put option
        """
        fwd = self.model.spot*self.model.mean(maturity)
        df = np.exp(-self.r*maturity)
        return self.call(strike, maturity) - df*(fwd - strike)

    def density(self, time, u):
        """Density function implied from the FFT method"""
        raise NotImplementedError("fft density function")
