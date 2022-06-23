"""
Transitions kernels
===================

Transition or perturbation strategies to propose new parameters based on
the current population.
Usually this translates to randomly selecting a parameter in the current
generation and then perturbing it, but in general arbitrary transition
kernels are possible.

These can be passed to :class:`pyabc.smc.ABCSMC` via the ``transitions``
keyword argument.
"""

from .base import AggregatedTransition, DiscreteTransition, Transition
from .exceptions import NotEnoughParticles
from .grid_search import GridSearchCV
from .jump import DiscreteJumpTransition, PerturbationKernel
from .local_transition import LocalTransition
from .model import ModelPerturbationKernel
from .multivariatenormal import (
    MultivariateNormalTransition,
    scott_rule_of_thumb,
    silverman_rule_of_thumb,
)
from .randomwalk import DiscreteRandomWalkTransition
