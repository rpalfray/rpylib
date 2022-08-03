"""
Description of some standard financial derivatives:
   - Vanilla options
   - Digital
   - Barrier options
   - Lookback options
   - CDS
   - Bond
   - Swaption
   - etc

   and standard payoffs like:
       - Swap
       - Call
       - Put
       - Payer/Receiver
"""

from enum import Enum
from numbers import Real
from typing import Union

import numpy as np


class PayoffDates(Enum):
    """Payoff dates can be deterministic as it is the case for most of the financial products, but in the case
    of credit derivatives for example, cashflows are usually exchange were an underlying defaults and
    therefore the payoff dates are stochastic and depend on the precise path of the considered underlying.
    """
    DETERMINISTIC = 1
    STOCHASTIC = 2


class OptionType(Enum):
    """Option type"""
    FORWARD = 1
    VANILLA = 2
    DIGITAL = 3
    BARRIER = 4
    LOOKBACK = 5


class PayoffType(Enum):
    """Payoff type"""
    CALL = 1
    PUT = 2


class BarrierType(Enum):
    """Barrier type"""
    UP_AND_IN = 1
    UP_AND_OUT = 2
    DOWN_AND_IN = 3
    DOWN_AND_OUT = 4


class SwaptionType(Enum):
    """Swaption type"""
    RECEIVER = 1
    PAYER = 2


class Payoff:
    """A payoff is simply a function of the underlying which is the :func:`evaluate` in this class.
    """
    def __init__(self, payoff_dates_type: PayoffDates = PayoffDates.DETERMINISTIC):
        """
        :param payoff_dates_type: specify if the payoff dates are fixed (deterministic) or depends on the path of the
        underlying
        """
        self.payoff_dates_type = payoff_dates_type

    def __call__(self, *args, **kwargs) -> float:
        return self.evaluate(*args, **kwargs)

    def evaluate(self, underlying) -> float:
        """Return the valuation of the payoff"""
        raise NotImplementedError

    def process(self, times, path) -> None:
        """Process path and retrieve relevant information if needed

        :Example: in the case of a lookback option, one needs to retrieve the max(min) of the underlying over [0,T]
        """
        pass

    def dimension(self) -> int:
        """Payoff dimension. In most cases, the payoff is unidimensional. For a swaption, the dimension is the number
        of underlying swap rates in consideration."""
        return 1

    @staticmethod
    def create(option_type: OptionType, *args, **kwargs):
        """Factory method to create the payoff object
        :param option_type: option type
        :param args: additional parameters of the payoff
        :param kwargs: additional keyword parameters of the payoff
        """
        if option_type == OptionType.FORWARD:
            return Forward(*args, **kwargs)
        if option_type == OptionType.VANILLA:
            return Vanilla(*args, **kwargs)
        if option_type == OptionType.BARRIER:
            return Barrier(*args, **kwargs)
        if option_type == OptionType.LOOKBACK:
            return LookBack(*args, **kwargs)
        raise ValueError('option type not yet implemented')


class PayoffOnTheFly(Payoff):
    """This class is useful to create a payoff by passing the payoff function directly.

    .. note:: only non-path-dependent payoff are in the scope as the :func:`process` function is not overridden.
    """
    def __init__(self, function):
        """
        :param function: payoff function
        """
        super().__init__()
        self.function = function

    def evaluate(self, underlying) -> float:
        return self.function(underlying)


class FixedCoupon(Payoff):
    """Payoff that pays a fixed coupon"""
    def __init__(self, coupon: float):
        """
        :param coupon: value of the fixed coupon
        """
        super().__init__()
        self.coupon = coupon

    def evaluate(self, underlying: float) -> float:
        return self.coupon


class Forward(Payoff):
    """Forward contract: exchange of the underlying against the strike K"""
    def __init__(self, strike: float):
        super().__init__()
        self.strike = strike

    def evaluate(self, underlying: float) -> float:
        return underlying - self.strike


class Vanilla(Payoff):
    """Vanilla option, that is for the moment only call and put options.
    """
    def __init__(self, strike: Union[float, list[float]], payoff_type: PayoffType):
        """
        :param strike: option strike
        :param payoff_type: payoff type
        """
        super().__init__()
        self.strike = strike
        self.payoff_type = payoff_type

        if payoff_type == PayoffType.CALL:
            self._call_put = 1
        elif payoff_type == PayoffType.PUT:
            self._call_put = -1
        else:
            raise NotImplementedError('Vanilla payoff not yet implemented')

        if isinstance(strike, Real):
            self._dimension = 1
        else:
            self._dimension = len(strike)

    def dimension(self) -> int:
        return self._dimension

    def evaluate(self, underlying: float) -> float:
        return np.maximum(self._call_put*(underlying - self.strike), 0.0)


class CallSpread(Payoff):
    """Call spread: buy one call at strike K1 and one call at strike K2 with K1 < K2
    """
    def __init__(self, strike1: float, strike2: float):
        super().__init__()
        if not strike1 < strike2:
            raise ValueError('expected strike1 < strike2')
        self.strike1 = strike1
        self.strike2 = strike2

    def evaluate(self, underlying: float) -> float:
        if underlying > self.strike2:
            return self.strike2 - self.strike1
        return np.maximum(0.0, underlying - self.strike1)


class Butterfly(Payoff):
    """Buy one call at strike1, sell two call at strike2 and buy one call at strike3"""
    def __init__(self, strike1: float, strike2: float, strike3: float):
        super().__init__()
        if not strike1 < strike2 < strike3:
            raise ValueError('expected strike1 < strike2 < strike3')
        self.strikes = np.array([strike1, strike2, strike3])

    def evaluate(self, underlying: float) -> float:
        calls = np.maximum(0.0, underlying - self.strikes)
        return calls[0] - 2*calls[1] + calls[2]


class Digital(Payoff):
    """Payoff equal to:
        - call case: 1 if the underlying is greater than the strike, 0 otherwise
        - put case: 1 if the underlying is less than the strike, 0 otherwise
    """
    def __init__(self, strike: float, payoff_type: PayoffType):
        super().__init__()
        self.strike = strike
        self.payoff_type = payoff_type
        if payoff_type != PayoffType.CALL and payoff_type != PayoffType.PUT:
            raise ValueError('expected payoff_type to be CALL or PUT')

    def evaluate(self, underlying) -> float:
        is_above = underlying > self.strike
        call_digital = 1.0 if is_above else 0.0
        if self.payoff_type == PayoffType.CALL:
            return call_digital
        else:
            return 1 - call_digital


class Barrier(Payoff):
    """Barrier option: a call or put payoff is activated/deactivated (in/out) is the underlying goes
    above/below (up/down)"""
    def __init__(self, strike: float, payoff_type: PayoffType, barrier_type: BarrierType, barrier: float):
        """
        :param strike: strike of the option payoff
        :param payoff_type: payoff type
        :param barrier_type: barrier type: up-and-in, up-and-out, down-and-in or down-and-out
        :param barrier: barrier level of the payoff activation/deactivation
        """
        super().__init__()
        self.vanilla = Vanilla(strike=strike, payoff_type=payoff_type)
        self.barrier = barrier
        self.barrierEvent = False  # it might be True depending on the spot price
        self.barrierType = barrier_type

        # IN/OUT barrier
        if barrier_type in (BarrierType.DOWN_AND_IN, BarrierType.UP_AND_IN):
            self._evaluate_impl = self._evaluate_knockin
        else:
            self._evaluate_impl = self._evaluate_knockout

        # Down/Up barrier
        if barrier_type in (BarrierType.DOWN_AND_IN, BarrierType.DOWN_AND_OUT):
            self.process = self.__barrier_event_down
        else:
            self.process = self.__barrier_event_up

    def __barrier_event_down(self, _, path):
        for value in path:
            if value < self.barrier:
                self.barrierEvent = True
                break

    def __barrier_event_up(self, _, path):
        for value in path:
            if value > self.barrier:
                self.barrierEvent = True
                break

    def evaluate(self, underlying: float) -> float:
        return self._evaluate_impl(underlying)

    def _evaluate_knockout(self, underlying: float) -> float:
        """
        .. todo:: this needs to work when underlying is a np.array
        """
        return 0.0 if self.barrierEvent else self.vanilla.evaluate(underlying)

    def _evaluate_knockin(self, underlying: float) -> float:
        return self.vanilla.evaluate(underlying) if self.barrierEvent else 0.0


class LookBack(Payoff):
    """Lookback option, only put case implemented.
    The option pays (max(S) - prefixed_maximum)_+ where max(S) is the maximum of the underlying spot
    over the considered period.
    """
    def __init__(self, prefixed_maximum: float):
        """
        :param prefixed_maximum: prefixed maximum that is "strike" of the option
        """
        super().__init__()
        self.prefixed_maximum = prefixed_maximum
        self.max_spot = 0

    def process(self, times, path) -> None:
        self.max_spot = np.exp(path.max(axis=0))
        raise ValueError('it depends on the process representation')

    def evaluate(self, underlying: float) -> float:
        return np.maximum(0.0, self.max_spot - self.prefixed_maximum)


class Rainbow(Payoff):
    """A rainbow option pays a weighted average of performances, it is similar to an Asian option but with
    non-equal weight.
    """
    def __init__(self, weights: np.array, strike: float, payoff_type: PayoffType):
        """
        :param weights: list of weights to be applied to the performances, from the best one to the worst ones
        :param strike: option strike
        :param payoff_type: payoff type
        """
        super().__init__()
        self._weights = np.flip(np.array(weights))  # flip weights: from worst to best
        # FIXME check weight sum to 1.0
        self.strike = strike
        self._eps = None
        if payoff_type == PayoffType.CALL:
            self._eps = +1
        elif payoff_type == PayoffType.PUT:
            self._eps = -1
        else:
            raise ValueError('Payoff type not implemented for this rainbow option')

    def evaluate(self, underlying: np.array) -> float:
        sorted_underlying = np.sort(underlying)
        underlying_value = sum(self._weights*sorted_underlying)
        return max(0.0, self._eps*(underlying_value - self.strike))


class CDS(Payoff):
    """Credit default swap: at the time of default, the buyer of the contract receives the notional.

    .. note:: this formulation assumes that the payment of the spread is continuous in time.
    """
    def __init__(self, recovery_rate: float, spread: float, maturity: float, discounting):
        """
        :param recovery_rate: CDS recovery rate
        :param spread: CDS spread
        :param maturity: CDS maturity
        :param discounting: discounting function
        """
        super().__init__(payoff_dates_type=PayoffDates.STOCHASTIC)
        self.recovery_rate = recovery_rate
        self.spread = spread
        self._df = discounting  # discounting function
        self._T = maturity
        self._r = -np.log(discounting(1))
        self._df_T = discounting(maturity)

    def process(self, times, path) -> None:
        pass

    def evaluate(self, default_time) -> np.array:
        default_leg = 0 if default_time > self._T else (1 - self.recovery_rate)*self._df(default_time)
        fixed_leg = self.spread*(1 - self._df(min(self._T, default_time)))/self._r
        # small trick as payoffs are already discounted in the MC engine
        dl = default_leg/self._df_T
        fl = fixed_leg/self._df_T
        pv = dl - fl
        return pv


class Bond(Payoff):
    """The maturity of the bond is the expiry of the first Libor rate.

    .. note:: By design, the Lévy Libor model and the Lévy Forward model are defined in the terminal measure
              (with regard to the maturity of the last underlying rate) and, to keep it simple, the payoff is
              tweaked accordingly.
    """
    def __init__(self, underlying_rates: np.array, deltas: np.array):
        """
        :param underlying_rates: underlying rates values as of today
        :param deltas: accrual of the underlying rates
        """
        super().__init__()
        self.deltas = deltas
        self._factor = 1/np.prod(1 + deltas*underlying_rates)
        self._dimension = underlying_rates.size

    def evaluate(self, underlying_rates) -> float:
        res = np.prod(1 + self.deltas*underlying_rates)
        return res*self._factor


class Cap(Payoff):
    """
    The periods of the cap are defined by the inputs deltas periods

    .. note:: By design, the Lévy Libor model and the Lévy Forward model are defined in the terminal measure
              (with regard to the maturity of the last underlying rate) and, to keep it simple, the payoff is
              tweaked accordingly.
    """
    def __init__(self, underlying_rates: np.array, deltas: np.array, strike: float):
        """
        :param underlying_rates: underlying rates values as of today
        :param deltas: accrual of the underlying rates
        :param strike: strike of the cap
        """
        super().__init__()
        self.deltas = deltas
        self.strike = strike
        self._factor = 1/np.prod(1 + deltas*underlying_rates)
        self._dimension = underlying_rates.size

    def evaluate(self, underlying_rates) -> float:
        adj = np.cumprod(1 + self.deltas * underlying_rates)
        res = self.deltas*np.maximum(underlying_rates - self.strike, 0)*adj[::-1]
        payoff = np.sum(res)
        return payoff*self._factor


class Ratchet(Payoff):
    """
    The periods of the ratchet are defined by the inputs deltas periods.
        - The client pays a coupon :math:`c_i = min(max(H_i + Y, c_{i-1}), c_{i-1})`
          where :math:`c_{i-1}` is the previous coupon, :math:`Y` is the increment and
          :math:`H_i = \\tau_i * (L_i + spread)` with:

             * :math:`\\tau_i` the accrual period
             * :math:`L_i` the rate for the period :math:`[T_{i-1}, T_i]`
        - The client receives a funding leg with coupon :math:`r_i = gearing*L_i + margin`.

    .. note:: By design, the Lévy Libor model and the Lévy Forward model are defined in the terminal measure
              (with regard to the maturity of the last underlying rate) and, to keep it simple, the payoff is
              tweaked accordingly.
    """
    def __init__(self, deltas: np.array, funding_gearing: float, funding_margin: float,
                 structured_spread: float, structured_increment: float, first_rate: float):
        """
        :param deltas: accrual of the underlying rates
        :param funding_gearing: gearing of the funding leg
        :param funding_margin: margin of the funding leg
        :param structured_spread: spread of the structured leg
        :param structured_increment: increment Y of the structured leg
        :param first_rate: value of the first underlying rate
        """
        super().__init__()
        self.deltas = deltas
        self.gearing = funding_gearing
        self.margin = funding_margin
        self.spread = structured_spread
        self.increment = structured_increment
        self.first_rate = first_rate

    def evaluate(self, underlying_rates) -> float:
        adj = np.cumprod(1 + self.deltas * underlying_rates)
        c_previous = self.first_rate
        c = np.zeros_like(underlying_rates)
        for k, (libor, delta) in enumerate(zip(underlying_rates, self.deltas)):
            aux = delta*(libor + self.spread)
            c[k] = c_previous = min(max(aux, c_previous), c_previous + self.increment)

        funding_leg = self.deltas*(self.gearing*underlying_rates + self.margin)
        payoff = np.sum((c - funding_leg)*adj[::-1])
        return payoff


class Swaption(Payoff):
    """The expiry of the swaption is the expiry of the first Libor rate.

    .. note:: By design, the Lévy Libor model and the Lévy Forward model are defined in the terminal measure
              (with regard to the maturity of the last underlying rate) and, to keep it simple, the payoff is
              tweaked accordingly.
    """
    def __init__(self, underlying_rates: np.array, deltas: np.array, strike: np.array,
                 swaption_type: SwaptionType = SwaptionType.RECEIVER):
        """
        :param underlying_rates: underlying rates values as of today
        :param deltas: accrual of the underlying rates
        :param strike: strike of the swaption
        :param swaption_type: swaption type, that is receiver or payer (receiver meaning that the swaption is an option
        into entering a receiver swap that is receiving the fixed coupon and paying the floating leg)
        """
        super().__init__()
        self.deltas = deltas
        self.strike = strike
        self.type = swaption_type
        self._eps = 1 if swaption_type == SwaptionType.PAYER else -1
        self._factor = 1/np.prod(1 + deltas*underlying_rates)

    def evaluate(self, underlying_rates) -> float:
        aux = np.cumprod(1 + self.deltas*underlying_rates)
        payer_payoff = aux[-1] - 1 - self.strike*np.sum(self.deltas*aux[::-1])
        payoff = np.maximum(self._eps*payer_payoff, 0.0)
        return payoff*self._factor
