"""Handling of parameters with constraint, for example positivity, bounded, etc.

This is done by specifying the setter/getter properties of the parameter.
"""

from functools import partial


def argument_with_condition(argument_name, condition, message):
    def sp_getter(instance):
        return instance.__dict__[argument_name]

    def sp_setter(instance, value):
        if condition(value):
            instance.__dict__[argument_name] = value
        else:
            raise ValueError(message)

    return property(sp_getter, sp_setter)


positive = partial(
    argument_with_condition,
    condition=lambda x: x >= 0,
    message="expected a positive value",
)
negative = partial(
    argument_with_condition,
    condition=lambda x: x <= 0,
    message="expected a negative value",
)

strictly_positive = partial(
    argument_with_condition,
    condition=lambda x: x > 0,
    message="expected a strictly positive value",
)
strictly_negative = partial(
    argument_with_condition,
    condition=lambda x: x < 0,
    message="expected a strictly negative value",
)


def greater_than(value):
    return partial(
        argument_with_condition,
        condition=lambda x: x >= value,
        message="expected a value greater than " + str(value),
    )


def strictly_greater_than(value):
    return partial(
        argument_with_condition,
        condition=lambda x: x > value,
        message="expected a value strictly greater than " + str(value),
    )


def less_than(value):
    return partial(
        argument_with_condition,
        condition=lambda x: x <= value,
        message="expected a value less than " + str(value),
    )


def strictly_less_than(value):
    return partial(
        argument_with_condition,
        condition=lambda x: x < value,
        message="expected a value strictly greater than " + str(value),
    )


def between(left_bound, right_bound):
    return partial(
        argument_with_condition,
        condition=lambda x: left_bound <= x <= right_bound,
        message="expected a value between "
        + str(left_bound)
        + " and "
        + str(right_bound),
    )


def strictly_between(left_bound, right_bound):
    return partial(
        argument_with_condition,
        condition=lambda x: left_bound < x < right_bound,
        message="expected a value strictly between "
        + str(left_bound)
        + " and "
        + str(right_bound),
    )


positive_sequence = partial(
    argument_with_condition,
    condition=lambda x: all(xi >= 0 for xi in x),
    message="expected a sequence of positive elements",
)
negative_sequence = partial(
    argument_with_condition,
    condition=lambda x: all(xi <= 0 for xi in x),
    message="expected a sequence of negative elements",
)

strictly_positive_sequence = partial(
    argument_with_condition,
    condition=lambda x: all(xi > 0 for xi in x),
    message="expected of strictly sequence elements",
)
strictly_negative_sequence = partial(
    argument_with_condition,
    condition=lambda x: all(xi < 0 for xi in x),
    message="expected of strictly sequence elements",
)
