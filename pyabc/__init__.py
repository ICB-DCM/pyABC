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
from .distance_functions import (
    DistanceFunction,
    NoDistance,
    SimpleFunctionDistance,
    PNormDistance,
    AdaptivePNormDistance,
    ZScoreDistanceFunction,
    PCADistanceFunction,
    MinMaxDistanceFunction,
    PercentileDistanceFunction,
    RangeEstimatorDistanceFunction,
    DistanceFunctionWithMeasureList)
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
    SimpleAcceptor,
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
from .populationstrategy import (
    AdaptivePopulationSize,
    ConstantPopulationSize)
from . import visualization
from .version import __version__  # noqa: F401


__all__ = [
    "ABCSMC",
    # distance
    "DistanceFunction",
    "NoDistance",
    "SimpleFunctionDistance",
    "PNormDistance",
    "AdaptivePNormDistance",
    "ZScoreDistanceFunction",
    "PCADistanceFunction",
    "MinMaxDistanceFunction",
    "PercentileDistanceFunction",
    "RangeEstimatorDistanceFunction",
    "DistanceFunctionWithMeasureList",
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
    "SimpleAcceptor",
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
