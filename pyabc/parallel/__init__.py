"""
Parallel job execution
======================
"""

from .util import nr_cores_available
from .execution_contexts import DefaultContext, ProfilingContext
from .sge import SGE
from .util import sge_available
from .sampler import (MappingSampler, SingleCoreSampler,
                      MulticoreParticleParallelSampler, DaskDistributedSampler,
                      MulticoreEvalParallelSampler)

__all__ = ["SGE", "sge_available", "nr_cores_available",
           "MappingSampler", "SingleCoreSampler",
           "MulticoreParticleParallelSampler", "DaskDistributedSampler",
           "DefaultContext",
           "ProfilingContext",
           "MulticoreEvalParallelSampler"]
