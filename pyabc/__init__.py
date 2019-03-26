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
                      ConstantEpsilon,
                      QuantileEpsilon,
                      MedianEpsilon,
                      ListEpsilon)
from .smc import ABCSMC
from .storage import History
from .acceptor import (Acceptor,
                       SimpleAcceptor,
                       accept_use_current_time,
                       accept_use_complete_history)
from .model import (Model,
                    SimpleModel,
                    ModelResult,
                    IntegratedModel)
from .transition import (MultivariateNormalTransition,
                         LocalTransition,
                         DiscreteRandomWalkTransition)
from .populationstrategy import (AdaptivePopulationSize,
                                 ConstantPopulationSize)
from .transition import GridSearchCV
from . import visualization
from .version import __version__  # noqa: F401

__all__ = [
    "ABCSMC",
    # Distance start
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
    # Distance end
    "Epsilon",
    "ConstantEpsilon",
    "ListEpsilon",
    "QuantileEpsilon",
    "MedianEpsilon",
    # random_variables start
    "RVBase",
    "RV",
    "RVDecorator",
    "LowerBoundDecorator",
    "Distribution",
    "ModelPerturbationKernel",
    # random_variables end
    "Parameter",
    "GridSearchCV",
    "ConstantPopulationSize",
    "AdaptivePopulationSize",
    "MultivariateNormalTransition",
    "LocalTransition",
    "DiscreteRandomWalkTransition",
    "Acceptor",
    "SimpleAcceptor",
    "accept_use_current_time",
    "accept_use_complete_history",
    "ModelResult",
    "Model",
    "SimpleModel",
    "IntegratedModel",
    "History",
    "visualization",
]


try:
    loglevel = os.environ["ABC_LOG_LEVEL"].upper()
except KeyError:
    loglevel = "INFO"

logging.basicConfig(level=loglevel)
