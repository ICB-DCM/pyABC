"""
Approximate Bayesian computation - Sequential Monte Carlo
=========================================================

ABCSMC algorithms for Bayesian model selection.
"""


import os
import logging


from .parameters import Parameter
from .random_variables import (
    Distribution,
    ModelPerturbationKernel,
    RV,
    RVBase,
    RVDecorator,
    LowerBoundDecorator)
from .distance import (
    Distance,
    NoDistance,
    IdentityFakeDistance,
    AcceptAllDistance,
    SimpleFunctionDistance,
    PNormDistance,
    AdaptivePNormDistance,
    ZScoreDistance,
    PCADistance,
    MinMaxDistance,
    PercentileDistance,
    RangeEstimatorDistance,
    DistanceWithMeasureList)
from .epsilon import (
    Epsilon,
    ConstantEpsilon,
    QuantileEpsilon,
    MedianEpsilon,
    ListEpsilon)
from .smc import ABCSMC
from .storage import History
from .acceptor import (
    Acceptor,
    SimpleFunctionAcceptor,
    accept_use_current_time,
    accept_use_complete_history)
from .model import (
    Model,
    SimpleModel,
    ModelResult,
    IntegratedModel)
from .transition import (
    MultivariateNormalTransition,
    LocalTransition,
    DiscreteRandomWalkTransition,
    GridSearchCV)
from .population import (
    Particle,
    Population)
from .populationstrategy import (
    AdaptivePopulationSize,
    ConstantPopulationSize)
from . import visualization
from .version import __version__  # noqa: F401


__all__ = [
    "ABCSMC",
    # distance
    "Distance",
    "NoDistance",
    "IdentityFakeDistance",
    "AcceptAllDistance",
    "SimpleFunctionDistance",
    "PNormDistance",
    "AdaptivePNormDistance",
    "ZScoreDistance",
    "PCADistance",
    "MinMaxDistance",
    "PercentileDistance",
    "RangeEstimatorDistance",
    "DistanceWithMeasureList",
    # epsilon
    "Epsilon",
    "ConstantEpsilon",
    "ListEpsilon",
    "QuantileEpsilon",
    "MedianEpsilon",
    # random variable
    "RVBase",
    "RV",
    "RVDecorator",
    "LowerBoundDecorator",
    "Distribution",
    "ModelPerturbationKernel",
    # parameter
    "Parameter",
    # population
    "Particle",
    "Population",
    # population size
    "ConstantPopulationSize",
    "AdaptivePopulationSize",
    # transition
    "MultivariateNormalTransition",
    "LocalTransition",
    "DiscreteRandomWalkTransition",
    "GridSearchCV",
    # acceptor
    "Acceptor",
    "SimpleFunctionAcceptor",
    "accept_use_current_time",
    "accept_use_complete_history",
    # model
    "ModelResult",
    "Model",
    "SimpleModel",
    "IntegratedModel",
    # history
    "History",
    # visualization
    "visualization",
]


try:
    loglevel = os.environ["ABC_LOG_LEVEL"].upper()
except KeyError:
    loglevel = "INFO"

logging.basicConfig(level=loglevel)
