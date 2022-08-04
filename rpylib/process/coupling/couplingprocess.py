from typing import Union

from .couplinglevycopula import CouplingProcessLevyCopula
from .couplingmarkovchain import CouplingMarkovChain
from .couplingsde import CouplingSDE

CouplingProcess = Union[CouplingMarkovChain, CouplingSDE, CouplingProcessLevyCopula]
# CouplingProcess is defined as a type, but it might be better to have an interface class.
