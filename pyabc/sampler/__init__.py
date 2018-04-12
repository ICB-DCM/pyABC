"""
Multi-core and Distributed Sampling
===================================

The choice of the sampler determines in which way parallelization is performed.
See also the `explanation of the samplers <sampler.html>`_.
"""

from .singlecore import SingleCoreSampler
from .mapping import MappingSampler
from .multicore import MulticoreParticleParallelSampler
from .base import Sample, Sampler
from .dask_sampler import DaskDistributedSampler
from .multicore_evaluation_parallel import MulticoreEvalParallelSampler
from .redis_eps import (RedisEvalParallelSampler,
                        RedisEvalParallelSamplerServerStarter)
from .concurrent_future import ConcurrentFutureSampler

__all__ = ["Sample",
           "Sampler",
           "SingleCoreSampler",
           "MulticoreParticleParallelSampler",
           "MappingSampler",
           "DaskDistributedSampler",
           "RedisEvalParallelSampler",
           "MulticoreEvalParallelSampler",
           "RedisEvalParallelSamplerServerStarter",
           "ConcurrentFutureSampler"]
