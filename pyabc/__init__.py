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
from .distance_functions import (DistanceFunction,
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
from .epsilon import (Epsilon,
                      NoEpsilon,
                      ConstantEpsilon,
                      QuantileEpsilon,
                      MedianEpsilon,
                      ListEpsilon)
from .smc import ABCSMC
from .storage import History
from .acceptor import (Acceptor,
                       SimpleAcceptor,
                       UniformAcceptor,
                       StochasticAcceptor)
from .model import (Model,
                    SimpleModel,
                    ModelResult,
                    IntegratedModel)
from .transition import (MultivariateNormalTransition,
                         LocalTransition)
from .populationstrategy import (AdaptivePopulationSize,
                                 ConstantPopulationSize)
from .transition import GridSearchCV
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
    "NoEpsilon",
    "ConstantEpsilon",
    "ListEpsilon",
    "QuantileEpsilon",
    "MedianEpsilon",
    # random_variables
    "RVBase",
    "RV",
    "RVDecorator",
    "LowerBoundDecorator",
    "Distribution",
    "ModelPerturbationKernel",
    # div
    "Parameter",
    "GridSearchCV",
    "ConstantPopulationSize",
    "AdaptivePopulationSize",
    "MultivariateNormalTransition",
    "LocalTransition",
    # acceptor
    "Acceptor",
    "SimpleAcceptor",
    "UniformAcceptor",
    "StochasticAcceptor",
    # model
    "ModelResult",
    "Model",
    "SimpleModel",
    "IntegratedModel",
    # history
    "History"
]


try:
    loglevel = os.environ["ABC_LOG_LEVEL"].upper()
except KeyError:
    loglevel = "INFO"

logging.basicConfig(level=loglevel)
