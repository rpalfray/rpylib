"""Definition of a grid object which will be need to define spatial grid as well as time grid.
"""

from collections.abc import Iterable
from functools import singledispatchmethod
from math import prod

import numpy as np

from ..tools.parameter import strictly_positive


class Coordinate1D:
    """Coordinate for an axis (one-dimensional grid)
    """
    __slots__ = ('value',)

    def __init__(self, coordinate: int):
        """
        :param coordinate: integer corresponding to the position on the axis
        """
        self.value = coordinate

    def __repr__(self):
        return 'Coordinate1D(' + str(self.value) + ')'

    def __neg__(self):
        return Coordinate1D(-self.value)

    def __add__(self, other: int):
        return Coordinate1D(self.value + other)

    def __sub__(self, other: int):
        return Coordinate1D(self.value - other)

    def __iter__(self):
        yield from [self.value]

    def __eq__(self, other):
        return self.value == other

    def __mul__(self, other):
        return Coordinate1D(self.value*other)

    def __rmul__(self, other):
        return self.__imul__(other)

    def __imul__(self, other):
        self.value *= other
        return self

    def __hash__(self):
        return self.value.__hash__()


class CoordinateND:
    """Coordinate for a n-dimensional grid
    """
    __slots__ = ('value',)

    def __init__(self, coordinates: Iterable[int]):
        """
        :param coordinates: list of integers corresponding to the coordinates (positions) on each axis
        """
        self.value = tuple(coordinates)

    def __repr__(self):
        n = str(len(self.value))
        return 'Coordinate' + n + 'D' + repr(self.value) + ''

    def __neg__(self):
        return CoordinateND((-u for u in self.value))

    def __add__(self, other: Iterable[int]):
        return CoordinateND((u + v for u, v in zip(self.value, other)))

    def __sub__(self, other: Iterable[int]):
        return CoordinateND((u - v for u, v in zip(self.value, other)))

    def __iter__(self):
        yield from self.value

    def __eq__(self, other):
        return all(u == v for u, v in zip(self.value, other))

    def __mul__(self, other):
        return CoordinateND((v*other for v in self.value))

    def __rmul__(self, other):
        return self.__imul__(other)

    def __imul__(self, other):
        self.value = tuple(val*other for val in self.value)
        return self

    def __getitem__(self, item):
        return self.value[item]

    def __hash__(self):
        return self.value.__hash__()


class Coordinates:
    """General Coordinates object which handles both one-dimensional and n-dimensional cases
    """
    def __new__(cls, coordinates):
        if isinstance(coordinates, Iterable):
            return CoordinateND(coordinates)
        else:
            return Coordinate1D(coordinates)


class Grid:
    """A grid is a set of axes, each of them being in the form of an interval [a_0,a_1,...,a_K],
    and i is the position of the i-th element a_i.

    :Example:
    For a 2d-grid specified by the axes [a_0,a_1,...,a_K] and [b_0,b_1,...,b_L], the point of coordinates (i,j)
    has value (a_i, b_j)
    """
    def __init__(self, axes: list[np.array]):
        if not isinstance(axes, list):
            raise ValueError('the axes input should be a list of np.arrays')

        self.axes = axes
        self.dimension = len(axes)

    def number_of_points(self):
        return prod(axe.size for axe in self.axes)

    @singledispatchmethod
    def __getitem__(self, coordinates) -> float:
        return self.axes[0][coordinates.value]

    @__getitem__.register
    def _(self, coordinates: CoordinateND) -> tuple[float]:
        return tuple(self.axes[k][c] for k, c in enumerate(coordinates))

    @singledispatchmethod
    def __setitem__(self, coordinates, value):
        self.axes[coordinates] = value

    @__setitem__.register
    def __setitem__(self, coordinates: CoordinateND, value):
        for k, (coordinate, val) in enumerate(zip(coordinates, value)):
            self.axes[k][coordinate] = val


class Uniform1DGrid:
    """Build a one-dimensional uniform axis

    :param start: start element of the axes
    :param end: end element of the axes (included)
    :param num: (strictly positive) number of points between start and end
    """
    num: int = strictly_positive('num')

    def __init__(self, start: float, end: float, num: int):
        if start > end:
            raise ValueError('expected start<end')

        self.start = float(start)
        self.end = float(end)
        self.num = int(num)

        grid, self.step = np.linspace(start=start, stop=end, num=self.num, endpoint=True, retstep=True, dtype=np.float)
        self.grid = grid

    def __len__(self):
        return self.grid.size

    def __getitem__(self, item):
        return self.grid[item]

    def __str__(self) -> str:
        return 'UniformGrid(start={}, end={}, step={})'.format(self.start, self.end, self.step)
