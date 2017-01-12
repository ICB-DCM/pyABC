"""
Parallel job execution
======================
"""

from .util import nr_cores_available
from .execution_contexts import DefaultContext, ProfilingContext
from .sge import SGE
from .util import sge_available
from .sampler import MappingSampler, SingleCoreSampler, MulticoreSampler

