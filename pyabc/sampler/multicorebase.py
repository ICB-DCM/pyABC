from .base import Sampler
from ..sge import nr_cores_available


class MultiCoreSampler(Sampler):
    """
    Multicore sampler base class. This sampler is not functional but provides
    the number of cores selection functionality used by all the multiprocessing
    samplers.
    """
    def __init__(self, n_procs=None):
        super().__init__()
        self._n_procs = n_procs

    @property
    def n_procs(self):
        if self._n_procs is not None:
            return self._n_procs
        return nr_cores_available()
