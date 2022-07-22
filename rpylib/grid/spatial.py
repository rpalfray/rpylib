"""Spatial Grid definitions.

In this module we define several types of spatial grids to be applied to the simulation of the CTMC scheme via
Monte-Carlo.
"""

from functools import singledispatchmethod
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize

from .grid import Grid, Coordinates, Coordinate1D, CoordinateND
from ..model.levymodel.levymodel import LevyModel, LevyMeasure
from ..model.levycopulamodel import LevyCopulaModel
from ..model.levydrivensde.levydrivensde import LevyDrivenSDEModel


class SpatialGrid(Grid):
    """Spatial Grid definition

    .. note:: this object expects the axis to be arrays with increasing elements and this check is not carried out
    for performance reason.

    """
    def __init__(self, axes: list[np.array]):
        super().__init__(axes=axes)


class CTMCGrid(SpatialGrid):
    """Grid object for the simulation of the CTMC (Continuous-Time Markov Chain) via Monte-Carlo
    """
    def __init__(self, h: float, origin_coordinate: int, axes: list[np.array]):
        """Build the CTMC grid:
        :param h: the hyper-cube of size h centered in the origin of the grid is taken off the spatial grid
        :param origin_coordinate: coordinate of the grid origin
        :param axes: list of axes
        """
        super().__init__(axes=axes)
        self.h = h
        self.origin = 0.0
        self.origin_coordinate = Coordinates(origin_coordinate)
        if (dimension := len(axes)) > 1:
            self.origin = tuple([0.0]*dimension)
            self.origin_coordinate = Coordinates([origin_coordinate] * dimension)
        self.truncations = [(axis[0], axis[-1]) for axis in axes]

    @singledispatchmethod
    def outside(self, coordinate) -> bool:
        return any(c < 0 or c > len(self.axes[k]) - 1 for k, c in enumerate(coordinate))

    @outside.register
    def _(self, coordinate: Coordinate1D) -> bool:
        return coordinate.value < 0 or coordinate.value > len(self.axes[0]) - 1

    @outside.register
    def _(self, coordinate: int) -> bool:
        return coordinate < 0 or coordinate > len(self.axes[0]) - 1

    @singledispatchmethod
    def left_point(self, coordinate) -> float:
        """:return: the point of the left side of x where x = axes[0][position] (single axis grid)
        if x is itself the first point of the axes, return x
        """
        return self.axes[0][max(0, coordinate - 1)]

    @left_point.register
    def _(self, coordinate: Coordinate1D) -> float:
        return self.axes[0][max(0, coordinate.value - 1)]

    @left_point.register
    def _(self, coordinate: CoordinateND) -> tuple[float]:
        return tuple(self.axes[k][max(0, c - 1)] for k, c in enumerate(coordinate))

    @singledispatchmethod
    def right_point(self, coordinate) -> float:
        """:return: the point of the right side of x where x = axes[0][position] (single axis grid)
        if x is itself the last point of the axes, return x
        """
        return self.axes[0][min(len(self.axes[0]) - 1, coordinate + 1)]

    @right_point.register
    def _(self, coordinate: Coordinate1D) -> float:
        return self.axes[0][min(len(self.axes[0]) - 1, coordinate.value + 1)]

    @right_point.register
    def _(self, coordinate: CoordinateND) -> tuple[float]:
        grid_length = len(self.axes[0])  # FIXME: fixed length across all axes
        return tuple(self.axes[k][min(grid_length - 1, c + 1)] for k, c in enumerate(coordinate))

    @singledispatchmethod
    def middle(self, xi: tuple[float], xip: tuple[float]) -> tuple[float]:
        """:return: the middle-point of [xi, xip] that is the point which coordinates are equal to the average
        of the coordinates of xi and xip"""
        return tuple(0.5*(x + xp) for x, xp in zip(xi, xip))

    @middle.register
    def _(self, xi: float, xip: float) -> float:
        return 0.5*(xi + xip)

    def refine(self) -> None:
        """refine the axes, i.e. add all the 'middle' point for each interval [x_i, x_{i+1}] of the axes
        """
        for kth_axis, axis in enumerate(self.axes):
            for k, (xi, xip) in enumerate(zip(axis, axis[1:])):
                xim = self.middle(xi, xip)
                axis = np.insert(axis, 2*k + 1, xim)

            self.axes[kth_axis] = axis

        self.h /= 2
        self.origin_coordinate *= 2

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.axes, self.axes, label='grid')
        fig.legend()
        plt.title('CTMC Spatial Grid')
        plt.grid(True)
        plt.show()


class CTMCUniformGrid(CTMCGrid):
    """Uniform grid for the CTMC, each axis has uniform spatial step of size h.
    """
    def __init__(self, h, model: Union[LevyModel, LevyCopulaModel], truncation_probability=0.99999):
        """Building the uniform grid for the CTMC. Each axis is uniform with spatial step h. The bounds of the grid
        are calculated via the model and the truncation probability, namely each (left/right) axis bound is chosen
        such that the ratio of the remaining and the total mass is less than the truncation probability.

        :param h: uniform spatial step of the grid
        :param model: Lévy model in scope
        :param truncation_probability: used to defined the bounds of the grid
        """
        l, r = compute_truncation(model=model, h=h, truncation_probability=truncation_probability)
        nb_of_points_left = int(abs(l)/h)
        nb_of_points_right = int(r/h)
        if nb_of_points_left + nb_of_points_right > 1e8:
            raise ValueError('the number of points is greater than 10M, choose a smaller value for the '
                             'truncation_probability or  greater value for h')
        axis_left = np.linspace(start=l, stop=-h, num=nb_of_points_left)
        axis_right = np.linspace(start=h, stop=r, num=nb_of_points_right)
        axis = np.concatenate((axis_left, [0.0], axis_right))
        pivot_position = axis_left.size
        super().__init__(h=h, origin_coordinate=pivot_position, axes=[axis] * model.dimension_model())

    @classmethod
    def create_from_fixed_nb_of_points(cls, h: float, nb_of_points: int, dimension: int = 1):
        """This constructor allows the user to define directly the number of points of the grid which also
        defines the boundaries of the grid.

        :param h: uniform spatial step h
        :param nb_of_points: number of points for each axis
        :param dimension: grid dimension
        """
        axis_right = np.array([k*h for k in range(1, nb_of_points//2 + 1)])
        axis_left = np.array([-x for x in axis_right[::-1]])
        axis = np.concatenate((axis_left, [0.0], axis_right))
        pivot_position = axis_left.size
        obj = cls.__new__(cls)
        # Don't forget to call any polymorphic base class initializers
        super(CTMCUniformGrid, obj).__init__(h=h, origin_coordinate=pivot_position, axes=[axis] * dimension)
        return obj


class CTMCGridProbabilityStep(CTMCGrid):
    """CTMC Grid where each spatial step is defined by its corresponding jump probability.
    """
    def __init__(self, h: float, model: LevyModel, minimum_probability_step: float = 0.05, dimension: int = 1):
        """Building the CTMC grid with probability steps
        :param h: spatial step h
        :param model: Lévy model in scope needed to compute the jump probability
        :param minimum_probability_step:
        :param dimension: grid dimension
        """
        axis_left = compute_left_axis(h=h, levy_measure=model.levy_triplet.nu,
                                      minimum_probability_step=minimum_probability_step)
        axis_right = compute_right_axis(h=h, levy_measure=model.levy_triplet.nu,
                                        minimum_probability_step=minimum_probability_step)

        axis = np.concatenate((axis_left, [0.0], axis_right))
        pivot_position = axis_left.size

        self.minimum_probability_step = minimum_probability_step
        self.levy_measure = model.levy_triplet.nu
        self.intensity_of_jumps = model.mass(-np.inf, -h/2) + model.mass(h/2, np.inf)
        super().__init__(h=h, origin_coordinate=pivot_position, axes=[axis] * dimension)

    def middle(self, xi: float, xip: float) -> float:
        """In that case, the middle point is defined in terms of probability, that the middle point is such that there
        is equal probability to jump to the left/right point.

        :param xi: left point
        :param xip: right point
        """
        if xi == 0:
            return self.h/2
        if xip == 0:
            return -self.h/2

        mass = self.levy_measure.integrate
        p = 0.5*mass(xi, xip)/self.intensity_of_jumps

        def to_call(right):
            res = mass(xi, right)
            val = res/self.intensity_of_jumps
            return val - p

        sol_right = scipy.optimize.root_scalar(to_call, bracket=[xi, xip], method='brentq', x0=0.5*(xi+xip), xtol=1e-10)
        sol = sol_right.root

        return sol


class CTMCGridGeometric(CTMCGrid):
    """CTMC grid where the spatial step is geometric starting from the origin of the grid.
    """
    def __init__(self, h: float, model: Union[LevyModel, LevyCopulaModel, LevyDrivenSDEModel],
                 nb_of_points_on_each_side: int = 2, truncation_probability: float = 0.99999):
        """Build the geometric grid for the CTMC. Considering only the right axis of a unidimensional grid, one gets:
        [0, h, alpha*h, alpha^2*h,...] where the last element is defined by the truncation probability and alpha is
        implied from this last quantity and the number of points.

        :param h: spatial step
        :param model: Lévy model, needed along the truncation probability to compute the boundaries
        :param nb_of_points_on_each_side: number of point for each left/right semi-axis
        :param truncation_probability: truncation probability
        """
        if nb_of_points_on_each_side < 2:
            raise ValueError('expected nb_of_points_on_each_side >= 2')
        l, r = compute_truncation(model=model, h=h, truncation_probability=truncation_probability)
        axis_right = np.geomspace(start=h, stop=r, num=nb_of_points_on_each_side)
        axis_left = np.geomspace(start=l, stop=-h, num=nb_of_points_on_each_side)
        axis = np.concatenate((axis_left, [0.0], axis_right))
        pivot_position = len(axis_left)
        super().__init__(h=h, origin_coordinate=pivot_position, axes=[axis] * model.dimension_model())

    @classmethod
    def create_with_bounds(cls, h: float, truncations: tuple[float, float], dimension: int,
                           nb_of_points_on_each_side: int = 2):
        """This constructor allows the user to pass the truncations (boundaries of the grid) directly

        :param h: spatial step
        :param truncations: truncation parameters
        :param dimension: grid dimension
        :param nb_of_points_on_each_side: number of points for each (left/right) semi-axis
        """
        if nb_of_points_on_each_side < 2:
            raise ValueError('expected nb_of_points_on_each_side >= 2')
        l, r = truncations
        axis_right = np.geomspace(start=h, stop=r, num=nb_of_points_on_each_side)
        axis_left = np.geomspace(start=l, stop=-h, num=nb_of_points_on_each_side)
        axis = np.concatenate((axis_left, [0.0], axis_right))
        pivot_position = len(axis_left)

        obj = cls.__new__(cls)
        super(CTMCGridGeometric, obj).__init__(h=h, origin_coordinate=pivot_position, axes=[axis] * dimension)
        return obj


class CTMCCredit(CTMCGrid):
    """In the case of the Garreau-Kercheval Credit Model (see "A Structural Jump Threshold Framework for Credit Risk"
    by Garreau and Kercheval), one only require to know whether the underlying is above a threshold, hence the CTMC can
    be greatly simplified to only account for negative big jumps.
    """
    def __init__(self, h: float, level_a: Union[float, list[float]], model: Union[LevyModel, LevyCopulaModel],
                 symmetric_grid: bool = True):
        l, r = compute_truncation(model=model, h=h)
        if isinstance(level_a, float) and level_a < l or isinstance(level_a, list) and any(a < l for a in level_a):
            raise ValueError('level a smaller than the last left point in the grid')

        if model.dimension_model() == 1:
            eps = min(abs(l - level_a)/2, abs(level_a + h)/2)
            axis = np.array([l, level_a - eps, level_a + eps, -h, 0, h, r])
            axes = [axis]
        else:
            axes = []
            for a in level_a:
                eps = min(abs(l - a)/2, abs(a + h)/2)
                if symmetric_grid:
                    # symmetric axes -> this is a current limitation in the code with the pairing function in Z^d
                    axis_values = [l, a - eps, a + eps, -h, 0, h, -a - eps, -a + eps, r]
                else:
                    axis_values = [l, a - eps, a + eps, -h, 0, h, r]
                axis = np.array(axis_values)
                axes.append(axis)
        pivot_position = 4
        if any(axis[pivot_position] != 0 for axis in axes):
            raise ValueError('CTMCCredit grid error: pivot position')

        super().__init__(h=h, origin_coordinate=pivot_position, axes=axes)


def compute_truncation(model: Union[LevyModel, LevyCopulaModel, LevyDrivenSDEModel], h: float,
                       truncation_probability: float = 0.99999):
    """Compute truncations of the grid given the truncation probability and the model in scope.

    :param model: Lévy model
    :param h: spatial step h
    :param truncation_probability: truncation probability
    """
    if model.dimension_model() == 1:
        if isinstance(model, LevyModel):
            levy_measure = model.levy_triplet.nu
        else:
            levy_measure = model.driver.levy_triplet.nu
        l, r = compute_truncation_helper(levy_measure=levy_measure, h=h, probability=truncation_probability)
    else:
        models = model.driver.models if isinstance(model, LevyDrivenSDEModel) else model.models
        l_r = [compute_truncation_helper(levy_measure=m.levy_triplet.nu, h=h, probability=truncation_probability)
               for m in models]
        l_r_unpack = list(zip(*l_r))
        l, r = min(l_r_unpack[0]), max(l_r_unpack[1])

    return l, r


def compute_truncation_helper(levy_measure: LevyMeasure, h: float, probability: float) -> tuple[float, float]:
    """Helper function: compute the truncations for a specific axis.

    :param levy_measure: levy measure of the model in scope
    :param h: spatial step h
    :param probability: truncation probability
    """
    grid = CTMCGrid(h=h, origin_coordinate=2, axes=[np.array([-3 * h, -h, 0, h, 3 * h])])
    h_left = grid.middle(grid.left_point(grid.origin_coordinate), 0.0)
    h_right = grid.middle(0.0, grid.right_point(grid.origin_coordinate))

    intensity_of_right_jumps = levy_measure.integrate(h_right, np.inf)
    intensity_of_left_jumps = levy_measure.integrate(-np.inf, h_left)

    def fun_right(truncation):
        res = levy_measure.integrate(h_right, truncation)
        mass = res/intensity_of_right_jumps
        return mass - probability

    def fun_left(truncation):
        res = levy_measure.integrate(truncation, h_left)
        mass = res/intensity_of_left_jumps
        return mass - probability

    sol_right = scipy.optimize.root_scalar(fun_right, bracket=[h_right, 100], method='brentq')
    sol_left = scipy.optimize.root_scalar(fun_left, bracket=[-100, h_left], method='brentq')

    return sol_left.root, sol_right.root


def compute_right_axis(h: float, levy_measure: LevyMeasure, minimum_probability_step: float):
    intensity_of_jumps = levy_measure.integrate(-np.inf, -h/2) + levy_measure.integrate(h/2, np.inf)
    intensity_of_right_jumps = levy_measure.integrate(h/2, np.inf)
    right_axis = np.array([0, h])

    def fun_right(left, p):
        def to_call(right):
            res = levy_measure.integrate(left, right)
            mass = res/intensity_of_jumps
            return mass - p
        return to_call

    start_left, start_right = 0, h
    middle_point = start_right - (start_right-start_left)/2

    while True:
        p_left = (intensity_of_right_jumps - levy_measure.integrate(h/2, middle_point))/intensity_of_jumps
        if p_left < minimum_probability_step/2:
            last_point = start_right + 2*(start_right - middle_point)
            right_axis = np.append(right_axis, last_point)
            break
        try:
            sol_middle = scipy.optimize.root_scalar(fun_right(start_right, minimum_probability_step/2),
                                                    bracket=[start_right, 100], method='brentq', xtol=1e-10)
            middle_point = sol_middle.root
            sol_right = scipy.optimize.root_scalar(fun_right(middle_point, minimum_probability_step/2),
                                                   bracket=[middle_point, 100], method='brentq', xtol=1e-10)
            start_left, start_right = start_right, sol_right.root
        except:
            delta = abs(middle_point - start_right)
            start_left, start_right = start_right, start_right + 2*delta
            middle_point = start_left + delta

        right_axis = np.append(right_axis, start_right)

    return right_axis[1:]


def compute_left_axis(h: float, levy_measure: LevyMeasure, minimum_probability_step: float):
    intensity_of_jumps = levy_measure.integrate(-np.inf, -h/2) + levy_measure.integrate(h/2, np.inf)
    intensity_of_left_jumps = levy_measure.integrate(-np.inf, -h/2)
    left_axis = np.array([-h, 0])

    def fun_left(right, p):
        def to_call(left):
            res = levy_measure.integrate(left, right)
            mass = res/intensity_of_jumps
            return mass - p
        return to_call

    start_left, start_right = -h, 0
    middle_point = start_right - (start_right-start_left)/2

    while True:
        p_left = (intensity_of_left_jumps - levy_measure.integrate(middle_point, -h/2))/intensity_of_jumps
        if p_left < minimum_probability_step/2:
            last_point = start_left - 2*(middle_point - start_left)
            left_axis = np.insert(left_axis, 0, last_point)
            break
        try:
            sol_middle = scipy.optimize.root_scalar(fun_left(start_left, minimum_probability_step/2),
                                                    bracket=[-100, start_left], method='brentq', xtol=1e-10)
            middle_point = sol_middle.root
            sol_left = scipy.optimize.root_scalar(fun_left(middle_point, minimum_probability_step/2),
                                                  bracket=[-100, middle_point], method='brentq', xtol=1e-10)
            start_left, start_right = sol_left.root, start_left
        except:
            delta = abs(start_right - middle_point)
            start_left, start_right = start_left - 2*delta, start_left
            middle_point = start_left + delta

        left_axis = np.insert(left_axis, 0, start_left)

    return left_axis[:-1]
