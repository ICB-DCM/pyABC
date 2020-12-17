"""
pyABC
=====

ABCSMC algorithms for Bayesian parameter inference and model selection.

.. note::
    pyABC allows to parallelize the sampling process via various samplers.
    If you want to also parallelize single model simulations, be careful that
    both levels of parallelization work together well.
    In particular, if the environment variable OMP_NUM_THREADS is not set,
    pyABC sets it to a default of 1. For multi-processed sampling (the
    default at least on linux systems), the flag PYABC_NUM_PROCS can be used to
    determine on how many jobs to parallelize the sampling.
"""


import logging
import os

from . import visualization
from .acceptor import (
    Acceptor,
    ScaledPDFNorm,
    SimpleFunctionAcceptor,
    StochasticAcceptor,
    UniformAcceptor,
    pdf_norm_from_kernel,
    pdf_norm_max_found,
)
from .distance import (
    AcceptAllDistance,
    AdaptiveAggregatedDistance,
    AdaptivePNormDistance,
    AggregatedDistance,
    BinomialKernel,
    Distance,
    DistanceWithMeasureList,
    IdentityFakeDistance,
    IndependentLaplaceKernel,
    IndependentNormalKernel,
    MinMaxDistance,
    NegativeBinomialKernel,
    NoDistance,
    NormalKernel,
    PCADistance,
    PercentileDistance,
    PNormDistance,
    PoissonKernel,
    RangeEstimatorDistance,
    SimpleFunctionDistance,
    StochasticKernel,
    ZScoreDistance,
)
from .epsilon import (
    AcceptanceRateScheme,
    ConstantEpsilon,
    DalyScheme,
    Epsilon,
    EssScheme,
    ExpDecayFixedIterScheme,
    ExpDecayFixedRatioScheme,
    FrielPettittScheme,
    ListEpsilon,
    ListTemperature,
    MedianEpsilon,
    NoEpsilon,
    PolynomialDecayFixedIterScheme,
    QuantileEpsilon,
    Temperature,
    TemperatureBase,
    TemperatureScheme,
)
from .inference import ABCSMC
from .model import IntegratedModel, Model, ModelResult, SimpleModel
from .parameters import Parameter
from .population import Particle, Population
from .populationstrategy import AdaptivePopulationSize, ConstantPopulationSize
from .random_variables import RV, Distribution, LowerBoundDecorator, RVBase, RVDecorator
from .sampler import (
    ConcurrentFutureSampler,
    DaskDistributedSampler,
    MappingSampler,
    MulticoreEvalParallelSampler,
    MulticoreParticleParallelSampler,
    RedisEvalParallelSampler,
    RedisEvalParallelSamplerServerStarter,
    RedisStaticSampler,
    RedisStaticSamplerServerStarter,
    SingleCoreSampler,
)
from .storage import History, create_sqlite_db_id
from .transition import (
    AggregatedTransition,
    DiscreteJumpTransition,
    DiscreteRandomWalkTransition,
    GridSearchCV,
    LocalTransition,
    ModelPerturbationKernel,
    MultivariateNormalTransition,
)
from .version import __version__  # noqa: F401

try:
    loglevel = os.environ["ABC_LOG_LEVEL"].upper()
except KeyError:
    loglevel = "INFO"

logging.basicConfig(level=loglevel)

if "OMP_NUM_THREADS" not in os.environ:
    os.environ["OMP_NUM_THREADS"] = "1"
