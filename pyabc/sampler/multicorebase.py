from .base import Sampler
from ..sge import nr_cores_available
from multiprocessing import ProcessError, Process, Queue
from queue import Empty
from typing import List


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


def get_if_worker_healthy(worker: List[Process], queue: Queue):
    """

    Parameters
    ----------
    worker: List[Process]
        List of worker processes which should be in a healthy state,
        i.e. either terminated with exit code 0 or
    queue: Queue
        A multiprocessing queue which is fed by the workers

    Returns
    -------

    item: An item from the queue

    """
    while all(worker.exitcode in [0, None] for worker in worker):
        try:
            item = queue.get(True, 1)
            return item
        except Empty:
            pass
    else:
        raise ProcessError("At least one worker is dead")
