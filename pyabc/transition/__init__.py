"""
Transitions (Perturbation kernels)
==================================

Perturbation strategies. The classes defined here transition the current
population to the next one. pyABC implements global and local transitions.
Proposals for the subsequent generation are generated from the current
generation density estimates of the current generations.
This is equivalent to perturbing randomly chosen particles.

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
