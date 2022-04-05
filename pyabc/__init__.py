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

import logging
import os

# isort: off

from .version import __version__

# isort: on

from . import settings, visualization
from .acceptor import (
    Acceptor,
    FunctionAcceptor,
    ScaledPDFNorm,
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
    FunctionDistance,
    FunctionKernel,
    IndependentLaplaceKernel,
    IndependentNormalKernel,
    InfoWeightedPNormDistance,
    MinMaxDistance,
    NegativeBinomialKernel,
    NoDistance,
    NormalKernel,
    PCADistance,
    PercentileDistance,
    PNormDistance,
    PoissonKernel,
    RangeEstimatorDistance,
    SlicedWassersteinDistance,
    StochasticKernel,
    WassersteinDistance,
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
    SilkOptimalEpsilon,
    Temperature,
    TemperatureBase,
    TemperatureScheme,
)
from .inference import ABCSMC
from .model import FunctionModel, IntegratedModel, Model, ModelResult
from .parameters import Parameter
from .population import Particle, Population, Sample
from .populationstrategy import AdaptivePopulationSize, ConstantPopulationSize
from .predictor import (
    GPKernelHandle,
    GPPredictor,
    HiddenLayerHandle,
    LassoPredictor,
    LinearPredictor,
    MLPPredictor,
    ModelSelectionPredictor,
    Predictor,
)
from .random_variables import (
    RV,
    Distribution,
    DistributionBase,
    LowerBoundDecorator,
    RVBase,
    RVDecorator,
)
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
    Sampler,
    SingleCoreSampler,
    nr_cores_available,
)
from .storage import History, create_sqlite_db_id
from .sumstat import IdentitySumstat, PredictorSumstat, Sumstat
from .transition import (
    AggregatedTransition,
    DiscreteJumpTransition,
    DiscreteRandomWalkTransition,
    GridSearchCV,
    LocalTransition,
    ModelPerturbationKernel,
    MultivariateNormalTransition,
)
from .util import EventIxs
from .weighted_statistics import (
    effective_sample_size,
    resample,
    resample_deterministic,
    weighted_mean,
    weighted_median,
    weighted_mse,
    weighted_quantile,
    weighted_rmse,
    weighted_std,
    weighted_var,
)

# Set log level
try:
    loglevel = os.environ['ABC_LOG_LEVEL'].upper()
except KeyError:
    loglevel = 'INFO'

logger = logging.getLogger("ABC")
logger.setLevel(loglevel)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(name)s %(levelname)s: %(message)s'))
logger.addHandler(sh)

# Set number of threads e.g. for numpy. as pyabc uses parallelization on its
#  own, this is a safer default.
if 'OMP_NUM_THREADS' not in os.environ:
    os.environ['OMP_NUM_THREADS'] = '1'
