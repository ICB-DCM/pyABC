"""
Approximate Bayesian computation - Sequential Monte Carlo
=========================================================

ABCSMC algorithms for Bayesian model selection.
"""

from .parameters import Parameter
from .random_variables import (Distribution,
                                   Kernel,
                                   ModelPerturbationKernel,
                                   RV,
                                   RVBase,
                                   RVDecorator,
                                   LowerBoundDecorator,
                                   MultivariateMultiTypeNormalDistribution,
                                   NonEmptyMultivariateMultiTypeNormalDistribution,
                                   EmptyMultivariateMultiTypeNormalDistribution,
                                   RVDecorator)
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
from .model import Model, SimpleModel, ModelResult
from .transition import MultivariateNormalTransition
from .populationstrategy import AdaptivePopulationStrategy, ConstantPopulationStrategy
from .transition import GridSearchCV

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
    "MultivariateMultiTypeNormalDistribution",
    "NonEmptyMultivariateMultiTypeNormalDistribution",
    "EmptyMultivariateMultiTypeNormalDistribution",
    "Kernel",
    "ModelPerturbationKernel",
    # random_variables end
    "SQLDataStore",
    "ABCLoader",
    "GridSearchCV"
]


import os
import logging
try:
    loglevel = os.environ["ABC_LOG_LEVEL"]
    numeric_loglevel = getattr(logging, loglevel.upper())
    logging.getLogger().setLevel(numeric_loglevel)
except KeyError:
    pass
