"""Build the C code for the Hormann Poisson generator"""

from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(
    name="Poison Algorithms",
    ext_modules=cythonize("hormann.pyx", annotate=True),
    include_dirs=[numpy.get_include()],
)
