"""
Transitions (Perturbation Kernels)
==================================

Perturbation strategies. The classes defined here transition the current
population to the next one. pyABC implements global and local transitions.
Proposals for the subsequent generation are generated from the current
generation density estimates of the current generations.
This is equivalent to perturbing randomly chosen particles.

These can be passed to :class:`pyabc.smc.ABCSMC` via the ``transitions``
keyword argument.
"""

from .base import Transition, DiscreteTransition
from .multivariatenormal import (MultivariateNormalTransition,
                                 silverman_rule_of_thumb,
                                 scott_rule_of_thumb)
from .exceptions import NotEnoughParticles
from .model_selection import GridSearchCV
from .local_transition import LocalTransition
from .randomwalk import DiscreteRandomWalkTransition

__all__ = [
    "Transition",
    "DiscreteTransition",
    "MultivariateNormalTransition",
    "GridSearchCV",
    "NotEnoughParticles",
    "LocalTransition",
    "scott_rule_of_thumb",
    "silverman_rule_of_thumb",
    "DiscreteRandomWalkTransition",
]
