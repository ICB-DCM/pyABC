"""
pyABC
=====

ABC algorithms for likelihood-free Bayesian parameter inference and model
selection.

.. note::
    pyABC allows to parallelize the sampling process via various samplers.
    If you want to also parallelize single model simulations, be careful that
    both levels of parallelization work together well.
    In particular, if the environment variable OMP_NUM_THREADS is not set,
    pyABC sets it to a default of 1. For multi-processed sampling (the
    default at least on linux systems), the flag PYABC_NUM_PROCS can be used to
    determine on how many jobs to parallelize the sampling.
"""

from .version import __version__  # noqa: F401
from .parameters import Parameter
from .random_variables import (
    Distribution,
    RV,
    RVBase,
    RVDecorator,
    LowerBoundDecorator)
from .distance import (
    Distance,
    NoDistance,
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
    RedisStaticSampler,
    RedisEvalParallelSamplerServerStarter,
    RedisStaticSamplerServerStarter,
    MulticoreEvalParallelSampler,
    ConcurrentFutureSampler)
from .inference import ABCSMC
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
    GridSearchCV,
    AggregatedTransition,
    DiscreteJumpTransition,
    ModelPerturbationKernel)
from .population import (
    Particle,
    Population)
from .populationstrategy import (
    AdaptivePopulationSize,
    ConstantPopulationSize)
from . import visualization
from . import settings

import os
import logging

# Set log level
try:
    loglevel = os.environ['ABC_LOG_LEVEL'].upper()
except KeyError:
    loglevel = 'INFO'

logger = logging.getLogger("ABC")
logger.setLevel(loglevel)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter(
    '%(name)s %(levelname)s: %(message)s'))
logger.addHandler(sh)

# Set number of threads e.g. for numpy. as pyabc uses parallelization on its
#  own, this is a safer default.
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'
