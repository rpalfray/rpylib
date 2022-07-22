from typing import Union

from .couplingmarkovchain import CouplingMarkovChain
from .couplingsde import CouplingSDE
from .couplinglevycopula import CouplingProcessLevyCopula


CouplingProcess = Union[CouplingMarkovChain, CouplingSDE, CouplingProcessLevyCopula]
# CouplingProcess is defined as a type, but it might be better to have an interface class.
