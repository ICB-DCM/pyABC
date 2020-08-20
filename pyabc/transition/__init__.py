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

from .base import Transition, DiscreteTransition, AggregatedTransition
from .multivariatenormal import (MultivariateNormalTransition,
                                 silverman_rule_of_thumb,
                                 scott_rule_of_thumb)
from .exceptions import NotEnoughParticles
from .grid_search import GridSearchCV
from .local_transition import LocalTransition
from .randomwalk import DiscreteRandomWalkTransition
from .jump import PerturbationKernel, DiscreteJumpTransition
from .model import ModelPerturbationKernel
