"""
Approximate Bayesian computation - Sequential Monte Carlo
=========================================================

ABCSMC algorithms for Bayesian model selection.
"""
import os
import logging
from .parameters import Parameter
from .random_variables import (Distribution,
                               ModelPerturbationKernel,
                               RV,
                               RVBase,
                               RVDecorator,
                               LowerBoundDecorator)
from .distance_functions import (ZScoreDistanceFunction,
                                 PCADistanceFunction,
                                 MinMaxDistanceFunction,
                                 PercentileDistanceFunction,
                                 DistanceFunction,
                                 RangeEstimatorDistanceFunction,
                                 DistanceFunctionWithMeasureList)
from .epsilon import Epsilon, ConstantEpsilon, MedianEpsilon, ListEpsilon
from .smc import ABCSMC
from .storage import History
from .model import Model, SimpleModel, ModelResult, IntegratedModel
from .transition import MultivariateNormalTransition, LocalTransition
from .populationstrategy import (AdaptivePopulationStrategy,
                                 ConstantPopulationStrategy)
from .transition import GridSearchCV
from .version import __version__  # noqa: F401

__all__ = [
    "ABCSMC",
    # Distance start
    "DistanceFunction",
    "DistanceFunctionWithMeasureList",
    "ZScoreDistanceFunction",
    "PCADistanceFunction",
    "RangeEstimatorDistanceFunction",
    "MinMaxDistanceFunction",
    "PercentileDistanceFunction",
    # Distance end
    "Epsilon", "ConstantEpsilon", "ListEpsilon", "MedianEpsilon",
    # random_variables start
    "RVBase",
    "RV",
    "RVDecorator",
    "LowerBoundDecorator",
    "Parameter",
    "Distribution",
    "ModelPerturbationKernel",
    # random_variables end
    "SQLDataStore",
    "ABCLoader",
    "GridSearchCV",
    "ConstantPopulationStrategy",
    "AdaptivePopulationStrategy",
    "MultivariateNormalTransition",
    "LocalTransition",
    "SimpleModel",
    "ModelResult",
    "Model",
    "IntegratedModel",
    "History"
]


try:
    loglevel = os.environ["ABC_LOG_LEVEL"].upper()
except KeyError:
    loglevel = "INFO"

logging.basicConfig(level=loglevel)
