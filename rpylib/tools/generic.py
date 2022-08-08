"""Generic helpers for Python code
"""

from itertools import accumulate
from math import floor
from operator import mul

from gmpy2 import qdiv


def lazy_indices_product(args: list[int]):
    """itertools.product can blow up the memory because:
     'Before `product()` runs, it completely consumes the input iterables, keeping pools of values in memory to
     generate the products.' (-> https://docs.python.org/3/library/itertools.html#itertools.product).
     There is sometimes a need for a 'lazy' cartesian product. The following code is inspired from:
     http://phrogz.net/lazy-cartesian-product and we are also following the advice in
     (https://hackernoon.com/generating-the-nth-cartesian-product-e48db41bed3f) and we are using gmpy2 (not bigfloat)
     for floating-point division.

    This function generate all the possible tuples of indices (i1, i2,...) for i1 in range(l1), i2 in range(l2),...
    where l1, l2,... are the args passed to the function

    :return: 'lazy' cartesian product
    """
    moduli = args
    denominators = [1] + list(accumulate(reversed(moduli[1:]), mul))
    nb_of_elements = denominators[-1] * args[0]

    for n in range(nb_of_elements):
        yield tuple(
            int(floor(qdiv(n, denominators[k])) % moduli[k]) for k, _ in enumerate(args)
        )
