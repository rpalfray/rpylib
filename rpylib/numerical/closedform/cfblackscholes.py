"""
Closed-Form pricing formulas for the Black-Scholes model
"""

import numpy as np
from scipy.stats import norm


class CFBlackScholes:
    """Some Closed-form formulas for the Black-Scholes model"""
    
    eps = 1e-8  # used as a threshold for very low variance or maturity
    
    def __init__(self, bs_model: 'BlackScholesModel'):
        """
        :param bs_model: the Black-Scholes model object
        """
        import rpylib.model.levymodel.mixed.blackscholes
        if not isinstance(bs_model, rpylib.model.levymodel.mixed.blackscholes.BlackScholesModel):
            raise ValueError('expected a Black-Scholes model')
            
        self.bs_model = bs_model

    def forward(self, strike, maturity):
        """Price a forward in the Black-Scholes model

        :param strike: strike of the forward contract
        :param maturity: maturity of the forward contract

        Returns:
            The value of the forward contract
        """
        r, d, spot = self.bs_model.r, self.bs_model.d, self.bs_model.spot
        return spot*np.exp(-d*maturity) - strike*np.exp(-r*maturity)

    def _call_put(self, flag: int, strike: float, maturity: float):
        r, d, = self.bs_model.r, self.bs_model.d
        spot, sigma = self.bs_model.spot, self.bs_model.parameters.sigma

        df = np.exp(-r * maturity)
        fwd = spot * np.exp((r - d) * maturity)

        if sigma < CFBlackScholes.eps or spot < CFBlackScholes.eps or maturity < CFBlackScholes.eps:
            intrinsic = max(0.0, flag * (fwd - strike))
            return df * intrinsic

        stddev = sigma * np.sqrt(maturity)

        d1 = np.log(fwd / strike) / stddev + 0.5 * stddev
        d2 = d1 - stddev

        return df * flag * (fwd * norm.cdf(d1 * flag) - strike * norm.cdf(d2 * flag))

    def call(self, strike: float, maturity: float):
        """Price a Call option in the Black-Scholes model

        :param strike: call strike
        :param maturity: call maturity
        """
        return self._call_put(1, strike, maturity)
    
    def put(self, strike: float, maturity: float):
        """Price a Put option in the Black-Scholes model

        :param strike: put strike
        :param maturity: put maturity
        """
        return self._call_put(-1, strike, maturity)
    
    def butterfly(self, strike1: float, strike2: float, strike3: float, maturity: float):
        """Price a Butterfly option in the Black-Scholes model

        strike1: first strike
        strike2: second strike
        strike3: third strike
        maturity: butterfly maturity
        .. note:: we expect strike1 < strike2 < strike3
        """
        return self.call(strike1, maturity) - 2*self.call(strike2, maturity) + self.call(strike3, maturity)

    def digital(self, strike, maturity):
        """Price a digital option in the Black-Scholes model

        :param strike: digital strike
        :param maturity: digital maturity
        """
        r, d,  = self.bs_model.r, self.bs_model.d
        spot, sigma = self.bs_model.spot, self.bs_model.parameters.sigma
        fwd = spot*np.exp((r-d)*maturity)
        df = np.exp(-r*maturity)

        if sigma < CFBlackScholes.eps or spot < CFBlackScholes.eps or maturity < CFBlackScholes.eps:
            intrinsic = np.array([1 if fwd > k else 0 for k in strike])
            return df*intrinsic

        stddev = sigma*np.sqrt(maturity)
        d2 = np.log(fwd/strike)/stddev - 0.5*stddev

        return df*norm.cdf(d2)
