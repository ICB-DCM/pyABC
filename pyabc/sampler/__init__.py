"""
Parallel sampling
=================

Parallel multi-core and distributed sampling.

The choice of the sampler determines in which way parallelization is performed.
See also the `explanation of the samplers <sampler.html>`_.
"""

from .base import Sampler
from .concurrent_future import ConcurrentFutureSampler
from .dask_sampler import DaskDistributedSampler
from .mapping import MappingSampler
from .multicore import MulticoreParticleParallelSampler
from .multicore_evaluation_parallel import MulticoreEvalParallelSampler
from .multicorebase import nr_cores_available
from .redis_eps import (
    RedisEvalParallelSampler,
    RedisEvalParallelSamplerServerStarter,
    RedisStaticSampler,
    RedisStaticSamplerServerStarter,
)
from .singlecore import SingleCoreSampler
