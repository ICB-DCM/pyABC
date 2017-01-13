from multiprocessing import Process, Queue
from .base import Sampler
from .. import nr_cores_available
from .singlecore import SingleCoreSampler
import numpy as np
import random
import logging
from typing import Union

logger = logging.getLogger("MutlicoreSampler")

SENTINEL = None


def feed(feed_q, n_jobs, n_proc):
    for _ in range(n_jobs):
        feed_q.put(1)

    for _ in range(n_proc):
        feed_q.put(SENTINEL)


def work(feed_q, result_q, sample_one, simulate_one, accept_one):
    random.seed()
    np.random.seed()
    single_core_sampler = SingleCoreSampler()
    while True:
        arg = feed_q.get()
        if arg == SENTINEL:
            break
        res = single_core_sampler.sample_until_n_accepted(sample_one, simulate_one, accept_one, 1)
        result_q.put((res, single_core_sampler.nr_evaluations_))


class MulticoreSampler(Sampler):
    """
    Requires no pickling of the ``sample_one``, ``simulate_one`` and ``accept_one`` function.
    This is achieved using fork on linux.

    The simulation results are still pickled as they are transmitted
    from the worker processes back to the parent process.
    Depending on the kind of summary statistics this can be fast or slow.
    If your summary statistics are only a dict with a couple of numbers,
    the overhead should not be substantial.
    However, if your summary statistics are large numpy arrays
    or similar, this could cause overhad


    Parameters
    ----------
        n_procs: Union[int, None]
            If set to None, the Number of cores is determined according to
            :func:`parallel.util.nr_cores_available`.


    .. warning::

        Windows support is *not* tested.
        As there is no fork on Windows. This sampler might not work.

    """
    def __init__(self, n_procs: Union[int, None] = None):
        super().__init__()
        self.n_procs = n_procs if n_procs else nr_cores_available()

    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        logger.info("Start sampling on {} cores".format(self.n_procs))
        feed_q = Queue()
        result_q = Queue()

        feed_process = Process(target=feed, args=(feed_q, n, self.n_procs))

        worker_processes = [Process(target=work, args=(feed_q, result_q, sample_one, simulate_one, accept_one))
                            for _ in range(self.n_procs)]

        for proc in worker_processes:
            proc.start()

        feed_process.start()

        collected_results = []

        for _ in range(n):
            collected_results.append(result_q.get())

        feed_process.join()

        for proc in worker_processes:
            proc.join()

        # Queue's get close automatically on garbage collection
        # No explicit closing necessary.

        results, evaluations = zip(*collected_results)
        self.nr_evaluations_ = sum(evaluations)
        return sum(results, [])
