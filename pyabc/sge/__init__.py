"""

SGE
===

Parallel job execution on SGE like environments.

The functions and classes in the pyabc.sge package can be used for at least
two purposes:

1. The SGE.map method can be used together with the MappingSampler to
   parallelize ABC-SMC in a SGE/UGE infrastructure.
2. SGE.map can be used in a standalone mode to execute jobs on a SGE/UGE
   cluster. This is completely independent of ABC-SMC inference.

"""

from .execution_contexts import DefaultContext, NamedPrinter, ProfilingContext
from .sge import SGE
from .util import nr_cores_available, sge_available
