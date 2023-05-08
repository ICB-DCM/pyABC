"""
.. _api_sampler:

Parallel sampling
=================

Parallel multi-core and distributed sampling.

The choice of the sampler determines in which way parallelization is performed.
See also the `explanation of the samplers <sampler.html>`_.

.. note::
    pyABC allows to parallelize the sampling process via various samplers.
    If you want to also parallelize single model simulations, be careful that
    both levels of parallelization work together well.
    In particular, if the environment variable OMP_NUM_THREADS is not set,
    pyABC sets it to a default of 1. For multi-processed sampling (the
    default at least on linux systems), the flag PYABC_NUM_PROCS can be used to
    determine on how many jobs to parallelize the sampling.
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
