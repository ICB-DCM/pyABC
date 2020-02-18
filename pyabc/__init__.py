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
    AggregatedDistance,
    AdaptiveAggregatedDistance,
    ZScoreDistance,
    PCADistance,
    MinMaxDistance,
    PercentileDistance,
    RangeEstimatorDistance,
    DistanceWithMeasureList,
    StochasticKernel,
    NormalKernel,
    IndependentNormalKernel,
    IndependentLaplaceKernel,
    BinomialKernel,
    PoissonKernel,
    NegativeBinomialKernel)
from .epsilon import (
    Epsilon,
    NoEpsilon,
    ConstantEpsilon,
    QuantileEpsilon,
    MedianEpsilon,
    ListEpsilon,
    TemperatureBase,
    ListTemperature,
    Temperature,
    TemperatureScheme,
    AcceptanceRateScheme,
    ExpDecayFixedIterScheme,
    ExpDecayFixedRatioScheme,
    PolynomialDecayFixedIterScheme,
    DalyScheme,
    FrielPettittScheme,
    EssScheme)
from .sampler import (
    SingleCoreSampler,
    MulticoreParticleParallelSampler,
    MappingSampler,
    DaskDistributedSampler,
    RedisEvalParallelSampler,
    MulticoreEvalParallelSampler,
    ConcurrentFutureSampler)
from .smc import ABCSMC
from .storage import (
    History,
    create_sqlite_db_id)
from .acceptor import (
    Acceptor,
    SimpleFunctionAcceptor,
    UniformAcceptor,
    StochasticAcceptor,
    pdf_norm_from_kernel,
    pdf_norm_max_found,
    ScaledPDFNorm)
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
    # smc
    "ABCSMC",
    # distance
    "Distance",
    "NoDistance",
    "IdentityFakeDistance",
    "AcceptAllDistance",
    "SimpleFunctionDistance",
    "PNormDistance",
    "AdaptivePNormDistance",
    "AggregatedDistance",
    "AdaptiveAggregatedDistance",
    "ZScoreDistance",
    "PCADistance",
    "MinMaxDistance",
    "PercentileDistance",
    "RangeEstimatorDistance",
    "DistanceWithMeasureList",
    "StochasticKernel",
    "NormalKernel",
    "IndependentNormalKernel",
    "IndependentLaplaceKernel",
    "BinomialKernel",
    "PoissonKernel",
    "NegativeBinomialKernel",
    # epsilon
    "Epsilon",
    "NoEpsilon",
    "ConstantEpsilon",
    "ListEpsilon",
    "QuantileEpsilon",
    "MedianEpsilon",
    "TemperatureBase",
    "ListTemperature",
    "Temperature",
    "TemperatureScheme",
    "AcceptanceRateScheme",
    "ExpDecayFixedIterScheme",
    "ExpDecayFixedRatioScheme",
    "PolynomialDecayFixedIterScheme",
    "DalyScheme",
    "FrielPettittScheme",
    "EssScheme",
    # sampler
    "SingleCoreSampler",
    "MulticoreParticleParallelSampler",
    "MappingSampler",
    "DaskDistributedSampler",
    "RedisEvalParallelSampler",
    "MulticoreEvalParallelSampler",
    "ConcurrentFutureSampler",
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
    "UniformAcceptor",
    "StochasticAcceptor",
    "pdf_norm_from_kernel",
    "pdf_norm_max_found",
    "ScaledPDFNorm",
    # model
    "ModelResult",
    "Model",
    "SimpleModel",
    "IntegratedModel",
    # storage
    "History",
    "create_sqlite_db_id",
    # visualization
    "visualization",
]


try:
    loglevel = os.environ["ABC_LOG_LEVEL"].upper()
except KeyError:
    loglevel = "INFO"

logging.basicConfig(level=loglevel)
