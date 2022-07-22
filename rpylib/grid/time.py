"""Time grid definition, simply a one-dimensional grid, that is a simple axis, with increasing elements.

"""

from .grid import Uniform1DGrid


class TimeGrid(Uniform1DGrid):
    """Build a uniform time axis
    """
    def __init__(self, start: float, end: float, num: int = 2):
        """
        :param start: start time
        :param end: end time
        :param num: number of discretisation points, 2 by default (the start and end points)
        """
        if start < 0:
            raise ValueError('negative start in the time-axis is not allowed')

        super().__init__(start, end, num)

    def __str__(self) -> str:
        return 'Time' + Uniform1DGrid.__str__(self)

    def __mul__(self, other: float):
        return other*self.grid

    def __rmul__(self, other: float):
        return other*self.grid
