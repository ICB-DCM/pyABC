from .base import Sampler
from ..sge import nr_cores_available
from multiprocessing import ProcessError, Process, Queue
from queue import Empty
from typing import List


class MultiCoreSampler(Sampler):
    """
    Multi-core sampler base class. This sampler is not functional but provides
    the number of cores selection functionality used by all the multiprocessing
    samplers.

    Parameters
    ----------
    n_procs: int
        Number of processes.
    daemon: bool
        Whether to spawn workers in daemon mode.
    check_max_eval: bool
        Whether to check the maximum number of evaluations on the fly.
    """

    def __init__(self,
                 n_procs: int = None,
                 daemon: bool = True,
                 check_max_eval: bool = False):
        super().__init__()
        self._n_procs = n_procs
        self.daemon = daemon
        self.check_max_eval = check_max_eval

    @property
    def n_procs(self):
        if self._n_procs is not None:
            return self._n_procs
        return nr_cores_available()


def healthy(worker):
    return all(worker.exitcode in [0, None] for worker in worker)


def get_if_worker_healthy(workers: List[Process], queue: Queue):
    """

    Parameters
    ----------

    workers: List[Process]
        List of worker processes which should be in a healthy state,
        i.e. either terminated with exit code 0 (success) or are still
        running (exitcode is None in this case)

    queue: Queue
        A multiprocessing queue which is fed by the workers

    Returns
    -------

    item: An item from the queue

    """
    while True:
        try:
            item = queue.get(True, 5)
            return item
        except Empty:
            if not healthy(workers):
                raise ProcessError("At least one worker is dead.")
    raise Exception("The code should never reach here")
