"""Lévy Forward Model
"""

from typing import Union

import numpy as np

from .levydrivensde import LevyDrivenSDEModel, LevyDriver, ForwardMarketSDEFunction


class LevyForwardModel(LevyDrivenSDEModel):
    """Lévy Forward Model"""
    def __init__(self, ois_rates: Union[float, np.array], tenors: list[float], sigma: np.array, driver: LevyDriver):
        super().__init__(driver=driver, x0=np.array(ois_rates), a=ForwardMarketSDEFunction(sigma=sigma, tenors=tenors))
        self.tenors = np.array(sorted(tenors))
        self.sigma = sigma
        self.deltas = np.diff(tenors).astype(float)

        # consistency checks:
        if not driver.finite_first_moment():
            raise ValueError('In the Lévy model, the driver must have a finite first moment.')

        if len(tenors) != self._m + 1:
            raise ValueError('Find {} tenors instead of the expected {}(=size of x0)'.format(len(tenors)-1, self._m))

    def __repr__(self):
        return 'LevyForwardModel'  # TODO

    def truncate_levy_measure(self, truncations) -> None:
        self.driver.truncate_levy_measure(truncations)

    def df(self, t: float) -> float:
        pos = np.searchsorted(self.tenors, t)  # self.tenors[pos - 1] < t <= self.tenors[pos]
        if pos == 0:
            aux = 1 + self.x0[0] * t
        else:
            aux = 1 + self.x0[0] * self.tenors[0]  # constant rate over [0, T1]
            aux *= np.prod([1 + self.x0[k] * (self.tenors[k + 1] - self.tenors[k]) for k in range(pos - 1)])
            aux = 1 + self.x0[pos - 1] * (t - self.tenors[pos - 1])
        return 1 / aux
