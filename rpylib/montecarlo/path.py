"""Description of a simulated Monte-Carlo path
"""

import abc
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

import rpylib
from .configuration import Configuration, VarianceReduction
from .statistic.statistic import PT
from ..process.process import ProcessRepresentation
from ..product.product import ControlVariates, Product
from ..product.underlying import Spot


def create_path(
    configuration: Configuration, deterministic_path: Callable[[np.array], NDArray]
):
    """
    :param configuration: Monte-Carlo configuration
    :param deterministic_path: values of the deterministic underlying
    :return: the path manager
    """
    has_antithetic = configuration.variance_reduction.has(VarianceReduction.ANTITHETIC)

    if isinstance(configuration, rpylib.montecarlo.configuration.ConfigurationStandard):
        if has_antithetic:
            raise NotImplementedError

        return MCPath(
            deterministic_path=deterministic_path,
            activate_spot_underlying=configuration.activate_spot_statistics,
        )

    if isinstance(
        configuration, rpylib.montecarlo.configuration.ConfigurationMultiLevel
    ):
        if has_antithetic:
            raise NotImplementedError(
                "Antithetic method not yet implemented for Multilevel Monte-Carlo"
            )

        return MLMCPath(
            deterministic_path=deterministic_path,
            activate_spot_underlying=configuration.activate_spot_statistics,
        )

    raise NotImplementedError("create_path")


class StochasticPath:
    """General Stochastic path"""

    def value(self) -> np.array:
        """Value of the path at each simulation time"""
        raise NotImplementedError

    def times(self) -> np.array:
        """Simulation times"""
        raise NotImplementedError

    def antithetic_value(self) -> np.array:
        """ANTITHETIC values of the path"""
        raise NotImplementedError


class StochasticJumpPath(StochasticPath):
    """Stochastic path with a jump component

    .. todo:: find a better name?
    """

    __slots__ = ("jump_times", "diffusion_path", "jump_path")

    def __init__(
        self, jump_times: np.array, diffusion_path: np.array, jump_path: np.array
    ):
        """
        :param jump_times: jump times
        :param diffusion_path: pure diffusive component
        :param jump_path: pure jump component
        """
        self.jump_times = jump_times
        self.diffusion_path = diffusion_path
        self.jump_path = jump_path

    def value(self) -> np.array:
        return self.diffusion_path + self.jump_path

    def value_jump(self) -> np.array:
        """Only the jump path"""
        return self.jump_path

    def times(self) -> np.array:
        return self.jump_times

    def antithetic_value(self) -> np.array:
        # FIXME: We could consider -+ and -- too
        return self.diffusion_path - self.jump_path


class StochasticSDEPath(StochasticJumpPath):
    """Stochastic path for a process defined by a SDE"""

    __slots__ = "drift"

    def __init__(
        self,
        drift: np.array,
        jump_times: np.array,
        diffusion_path: np.array,
        jump_path: np.array,
    ):
        """
        :param drift: SDE drift values
        :param jump_times: jump times
        :param diffusion_path: pure diffusive component
        :param jump_path: pure jump component
        """
        super().__init__(jump_times, diffusion_path, jump_path)
        self.drift = drift

    def value(self) -> np.array:
        return self.drift + super().value()

    def antithetic_value(self) -> np.array:
        raise self.drift + super().antithetic_value()


class MCPath(abc.ABC):
    """MCPath handles the different ways of storing a (possibly multidimensional) Monte-Carlo path, and it is also
    responsible for computing the underlying and the final payoff values.
    """

    def __init__(
        self,
        deterministic_path: Callable[[np.array], np.array],
        activate_spot_underlying: bool,
    ):
        """
        :param deterministic_path: function that gives the deterministic value of the process
        :param activate_spot_underlying: if True then compute the modelled underlying too
        """
        self.payoff = None
        self.payoff_control_variates = 0.0

        # FIXME: we don't always need the spot underlying
        self.spot_underlying = 0
        self.spot = Spot()
        if not activate_spot_underlying:
            self.process_spot = lambda path: None

        self.deterministic_path = deterministic_path
        self.stochastic_path: StochasticJumpPath = (
            None  # this the stochastic part of the process
        )

    def update(self, process_representation: ProcessRepresentation):
        """Update the evaluation function with regard to the process representation"""
        self.spot.update(process_representation)

    def process_spot(self, path):
        """Process the spot underlying if spot statistics are required"""
        # 'times' is a dummy variable as Spot means the final spot value
        self.spot_underlying = self.spot.value(times=None, path=path, jump_path=None)

    def set_to_path(self, stochastic_path) -> None:
        """Set computed path"""
        self.stochastic_path = stochastic_path

    def process(self, product: Product, control_variates: ControlVariates) -> None:
        """Compute the payoff value from the simulated path"""
        times = self.stochastic_path.times()
        path = self.deterministic_path(times) + self.stochastic_path.value()
        jump_path = self.stochastic_path.value_jump()
        payoff_underlying = product.underlying_value(times, path, jump_path)
        self.process_spot(path)
        self.payoff = product(payoff_underlying)
        # add control variates -- FIXME: to be optimised -> the spot underlying is computed several times
        self.payoff_control_variates = control_variates.process(
            times, path, jump_path, payoff_underlying
        )

    def discount(self, df: float):
        """Discount the payoff with df
        :param df: discount
        """
        self.payoff *= df
        self.payoff_control_variates *= df

    def __repr__(self) -> str:
        return str(np.exp(self.deterministic_path + self.stochastic_path.value()))


class MLMCPath(MCPath):
    """Handling of the path for the Multilevel Monte-Carlo"""

    def __init__(
        self,
        deterministic_path: Callable[[np.array], np.array],
        activate_spot_underlying: bool,
    ):
        """
        :param deterministic_path: function that gives the deterministic value of the process
        :param activate_spot_underlying: if True then compute the modelled underlying too
        """
        super().__init__(deterministic_path, activate_spot_underlying)
        if not activate_spot_underlying:
            self.process_spot_level_l = lambda path_fine, path_coarse: None

    def process_spot_level_l(self, path_fine, path_coarse) -> None:
        """Processing the spot underlying for both the fine and coarse paths

        :param path_fine: path of the `fine` process
        :param path_coarse: path of the `coarse` process
        """
        spot_fine = self.spot.value(times=None, jump_path=None, path=path_fine)
        spot_coarse = self.spot.value(times=None, jump_path=None, path=path_coarse)
        self.spot_underlying = np.hstack(
            (spot_fine[np.newaxis].T, spot_coarse[np.newaxis].T)
        )

    def process(self, product: Product, control_variates: ControlVariates) -> None:
        """Processing the payoff for both the fine and coarse paths

        :param product: product to price
        :param control_variates: control variates
        """
        times = self.stochastic_path.times()
        path = self.deterministic_path(times) + self.stochastic_path.value()
        jump_path = self.stochastic_path.value_jump()
        # path = [path_fine, path_coarse] which might be a bit confusing because for the stats
        # the fine/coarse difference is given on the last dimension
        path_fine = path[PT.FP, ...]
        path_coarse = path[PT.CP, ...]
        jump_path_fine = jump_path[PT.FP, ...]
        jump_path_coarse = jump_path[PT.CP, ...]
        payoff_underlying_from_fp = product.underlying_value(
            times, path_fine, jump_path_fine
        )
        payoff_underlying_from_cp = product.underlying_value(
            times, path_coarse, jump_path_coarse
        )
        self.payoff = np.array(
            [product(payoff_underlying_from_fp), product(payoff_underlying_from_cp)]
        )
        self.process_spot_level_l(path_fine, path_coarse)
        self.payoff_control_variates = control_variates.process_mlmc(
            times,
            path_fine,
            path_coarse,
            jump_path_fine,
            jump_path_coarse,
            payoff_underlying_from_fp,
            payoff_underlying_from_cp,
        )

    def process_l0(self, product: Product, control_variates: ControlVariates) -> None:
        """Processing the payoff for the first level (=0) that is when there is no coarse process

        :param product: product to price
        :param control_variates: control variates
        """
        times = self.stochastic_path.times()
        path = self.deterministic_path(times) + self.stochastic_path.value()
        jump_path = self.stochastic_path.value_jump()
        payoff_underlying = product.underlying_value(times, path, jump_path)
        payoff = product(payoff_underlying)
        self.payoff = np.array([payoff, 0.0])
        self.process_spot(path)
        self.payoff_control_variates = control_variates.process(
            times=times,
            path=path,
            jump_path=jump_path,
            payoff_underlying=payoff_underlying,
        )
