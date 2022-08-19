"""This module describes a financial product which is defined as a payoff applied to an underlying

    Product are considered mono-underlying for the moment.
"""

import logging
from numbers import Real
from typing import Union

import numpy as np

from .payoff import Payoff
from .underlying import Underlying
from ..grid.time import TimeGrid
from ..montecarlo.statistic.statistic import PT
from ..process.process import ProcessRepresentation


class Product:
    """A financial product consists of a payoff function applied at maturity T to an underlying.

    .. note:: American/Bermudan-like products are not in scope for the moment
    """

    def __init__(
        self,
        payoff_underlying: Underlying,
        payoff: Payoff,
        maturity: float,
        notional: float = 1.0,
    ):
        """
        :param payoff_underlying: underlying
        :param payoff: payoff object
        :param maturity: product maturity
        :param notional: notional
        """
        self.payoff_underlying = payoff_underlying
        self.payoff = payoff
        self.maturity = maturity
        self.notional = notional

    def underlying_value(
        self, times: TimeGrid, path: np.array, jump_path: np.array
    ) -> float:
        """Compute the value of the payoff underlying from the underlying path.

        :param times: times of the underlying path
        :param path: underlying path values
        :param jump_path: only pure jump path of the underlying
        :return: the payoff underlying
        """
        underlying = self.payoff_underlying.value(
            times=times, path=path, jump_path=jump_path
        )
        self.payoff.process(times, path)
        return underlying

    def update(self, process_representation: ProcessRepresentation) -> None:
        """Update of the payoff underlying object given the process representation (identity or log-representation)

        :param process_representation: process representation in the modelling framework, that is either directly the
                                       underlying or the log-underlying
        """
        self.payoff_underlying.update(process_representation)

    def __call__(self, underlying) -> float:
        """Applies the underlying value to the payoff product.

        :param underlying: underlying value
        :return: the value of the payoff product for a value of the underlying
        """
        return self.notional * self.payoff(underlying)

    def times_grid(self) -> TimeGrid:
        """Compute the times grid of the product.

        :return: the times grid, that is the relevant dates needed to value the payoff
        """
        return self.payoff_underlying.compute_times_grid(maturity=self.maturity)


class ControlVariates:
    """Standard control variates object used to decrease the variance of the Monte-Carlo estimator.

    .. note:: by default, the control variates have the same maturity as the product in scope.
    """

    def __init__(self, products: [Product], prices: [Union[float, np.array]]):
        """Build the control variates from a list of products and their market prices.

        :param products: products used as control variates
        :param prices: market prices of these products
        """
        if len(products) != len(prices):
            raise ValueError(
                "wrong inputs in ControlVariates: payoffs and prices have different sizes"
            )
        self.products = products
        self.prices = prices
        self._underlying_functions = []
        self.nb_cvs = len(prices)

    def initialisation(self, payoff_underlying_type) -> None:
        """If two underlyings are closely related, for example S and log(S), one don't need to compute each of them as
        one can be implied from this other. This function aims to implement this logic.

        :param payoff_underlying_type: this the :func:`type()` of the payoff underlying object

            .. todo:: we need to `order` the underlying so that one can be computed from another or at least
                      we need some kind of hierarchy mapping to express this relationship.
        """
        self._underlying_functions = [
            p.payoff_underlying.imply_from_payoff_underlying(payoff_underlying_type)
            for p in self.products
        ]

    def process(
        self, times, path: np.array, jump_path: np.array, payoff_underlying
    ) -> np.array:
        """Process the path information: compute the payoff underlyings, then the payoff value.

        :param times: times of the path
        :param path: path of the underlyings
        :param jump_path: only the pure jump path of the underlying
        :param payoff_underlying: payoff underlying values
        :return: the value of the payoff for these underlyings path
        """
        payoff_underlyings = [
            fun(times, path, jump_path, payoff_underlying)
            for fun in self._underlying_functions
        ]
        payoffs = [
            product(value) for product, value in zip(self.products, payoff_underlyings)
        ]
        # the following idea is that we want a 2d-array even for a 1d-array
        # ndmin=2 will create a 2d array but the transposed version of what we want
        # in the case of a 2d-array input, the line below just transposes twice the array.
        # Definitively not optimal...

        return np.array(np.array(payoffs).T, ndmin=2).T

    def process_mlmc(
        self,
        times,
        path_fine,
        path_coarse,
        jump_path_fine,
        jump_path_coarse,
        payoff_underlying_from_FP,
        payoff_underlying_from_CP,
    ) -> np.array:
        """Process the path information in the case of the MLMC

        :param times: times of the path
        :param path_fine: path of the `fine` underlying
        :param path_coarse: path of the `coarse` underlying
        :param jump_path_fine: pure jump path of the `fine` underlying
        :param jump_path_coarse: pure jump path of the `coarse` underlying
        :param payoff_underlying_from_FP: payoff underlying values given by the `fine` underlying
        :param payoff_underlying_from_CP: payoff underlying values given by the `coarse` underlying
        :return: an array of both the `fine` and `coarse` payoffs

        .. seealso:: this is the MLMC version of :func:`process`
        """
        payoff_underlyings_from_fine = np.array(
            [
                fun(times, path_fine, jump_path_fine, payoff_underlying_from_FP)
                for fun in self._underlying_functions
            ]
        )
        payoff_underlyings_from_coarse = np.array(
            [
                fun(times, path_coarse, jump_path_coarse, payoff_underlying_from_CP)
                for fun in self._underlying_functions
            ]
        )
        payoffs_fine = np.array(
            [
                product(value)
                for product, value in zip(self.products, payoff_underlyings_from_fine)
            ]
        )
        payoffs_coarse = np.array(
            [
                product(value)
                for product, value in zip(self.products, payoff_underlyings_from_coarse)
            ]
        )

        return np.array(
            [
                np.array(np.array(payoffs_fine).T, ndmin=2),
                np.array(np.array(payoffs_coarse).T, ndmin=2),
            ]
        ).T

    @staticmethod
    def helper_compute_coefficients(x: np.array, y: np.array, prices: np.array):
        """Helper that computes the (optimal) coefficients of the control variates

        :param x: nd numpy array corresponding to the simulation of the control variates
        :param y: flat 1d numpy array corresponding to the simulation of the payoff
        :param prices: market prices of the control variates
        :return: flat 1d array of the adjusted payoff simulation with the control variates
        """
        covariance = np.cov(x.T, y.T, bias=True)
        sigma_x = covariance[0:-1, 0:-1]
        sigma_xy = covariance[0:-1, -1]
        try:
            if np.amin(np.absolute(sigma_x)) < 1e-12:
                b_star = np.zeros_like(sigma_xy)
            else:
                inv_sigma_x = np.linalg.inv(sigma_x)
                b_star = inv_sigma_x @ sigma_xy
        except np.linalg.LinAlgError:
            logging.log(
                level=logging.WARNING,
                msg="control variate not taken into account in this pass as the "
                "correlation matrix is not positive definite",
            )
            b_star = np.zeros_like(sigma_xy)

        # update statistics input
        cv_stats = y - np.dot(b_star, (x - prices).T)
        return cv_stats

    def compute_coefficients(self, statistics: "MCStatistics") -> None:
        """Compute the (optimal) coefficients of the control variates.

        These formulas can be found in `Monte-Carlo Methods in Financial Engineering` by Paul Glasserman - 4.1.2
        Multiple Controls p196

        :param statistics: statistics object which contains all the simulations of the payoff and  the control variates

            .. note:: this function updates directly the relevant `_payoff_statistics_with_cv` member
        """
        # Y_i(b) = Y_i - b(X_i - E[X])
        X = statistics._control_variates_statistics.stats
        Y = statistics._payoff_statistics.stats

        res = np.empty_like(Y)
        for k, (xx, yy) in enumerate(zip(X.T, Y.T)):
            if isinstance(self.prices[0], Real):  # FIXME: not the most elegant way...
                prices = self.prices[k]
            else:
                prices = np.array([elmt[k] for elmt in self.prices])
            cv_stats = self.helper_compute_coefficients(x=xx.T, y=yy.T, prices=prices)
            res[:, k] = cv_stats
        statistics._payoff_statistics_with_cv.stats = res

    def compute_coefficients_mlmc(self, statistics: "MCStatistics", level: int) -> None:
        """Compute the (optimal) coefficients of the control variates for the MLMC.

        :param statistics: statistics object
        :param level: current level in the Multilevel Monte-Carlo

            .. seealso:: function :func:`compute_coefficients`
        """
        # Y_i(b) = Y_i - b(X_i - E[X])
        X = statistics._control_variates_statistics.stats
        Y = statistics._payoff_statistics.stats

        if level == 0:
            # in that case -> control variates: the coarse process is not simulated, and we just set it to 0
            X_fine = X
            X_coarse = np.zeros_like(X_fine)
        else:
            # for level > 0, the shape of cv is (nb_mc_paths, nb_of_cvs, 2) -> 2 for the fine and coarse values
            X_fine = X[..., PT.FP]
            X_coarse = X[..., PT.CP]

        # the shape of payoff is (nb_mc_paths, 1, 2) in both cases (level=0 or level>0)
        Y_fine = Y[..., PT.FP]
        Y_coarse = Y[..., PT.CP]

        adj_payoff_fine = np.empty_like(Y_fine)
        adj_payoff_coarse = np.empty_like(Y_coarse)
        for k, (xx_fine, yy_fine, xx_coarse, yy_coarse) in enumerate(
            zip(X_fine.T, Y_fine.T, X_coarse.T, Y_coarse.T)
        ):
            if isinstance(self.prices[0], Real):
                prices = self.prices[k]
            else:
                prices = np.array([elmt[k] for elmt in self.prices])
            cv_stats_fine = self.helper_compute_coefficients(
                x=xx_fine.T, y=yy_fine.T, prices=prices
            )
            cv_stats_coarse = self.helper_compute_coefficients(
                x=xx_coarse.T, y=yy_coarse.T, prices=prices
            )
            adj_payoff_fine[:, k] = cv_stats_fine
            adj_payoff_coarse[:, k] = cv_stats_coarse

        cv_stats = np.zeros_like(statistics._payoff_statistics_with_cv.stats)
        cv_stats[..., PT.FP] = adj_payoff_fine
        cv_stats[..., PT.CP] = adj_payoff_coarse

        statistics._payoff_statistics_with_cv.stats = cv_stats


class NoControlVariates(ControlVariates):
    """Dummy class when there is no control variate"""

    def __init__(self):
        super().__init__(products=[], prices=[])

    def process(
        self, times, path: np.array, jump_path: np.array, payoff_underlying
    ) -> np.array:
        return 0.0

    def process_mlmc(
        self,
        times,
        path_fine,
        path_coarse,
        jump_path_fine,
        jump_path_coarse,
        payoff_underlying_from_FP,
        payoff_underlying_from_CP,
    ) -> np.array:
        return 0.0

    def initialisation(self, payoff_underlying_type) -> None:
        pass

    def compute_coefficients(self, statistics: "MCStatistics"):
        pass

    def compute_coefficients_mlmc(self, statistics: "MCStatistics", level: int):
        pass
