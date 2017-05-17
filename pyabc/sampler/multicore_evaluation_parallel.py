from multiprocessing import Process, Queue, Value
from ctypes import c_longlong
from .multicorebase import MultiCoreSampler
from ..sge import nr_cores_available
import numpy as np
import random


def work(sample, simulate, accept,
         queue, n_eval: Value, n_particles: Value):
    random.seed()
    np.random.seed()

    while n_particles.value > 0:
        with n_eval.get_lock():
            particle_id = n_eval.value
            n_eval.value += 1

        new_param = sample()
        new_sim = simulate(new_param)

        if accept(new_sim):
            with n_particles.get_lock():
                n_particles.value -= 1

            queue.put((particle_id, new_sim))


class MulticoreEvalParallelSampler(MultiCoreSampler):
    """
    Multicore Evaluation parallel sampler.

    Implements the same strategy as
    :class:`pyabc.sampler.RedisEvalParallelSampler`
    or
    :class:`pyabc.sampler.DaskDistributedSampler`.

    However, parallelization is restricted to a single machine with multiple
    processes.
    This sampler has very low communication overhead and is thus suitable
    for short running model evaluations.

    Requires no pickling of the ``sample_one``,
    ``simulate_one`` and ``accept_one`` function.
    This is achieved using fork on linux (see :class:`Sampler`).

    The simulation results are still pickled as they are transmitted
    from the worker processes back to the parent process.
    Depending on the kind of summary statistics this can be fast or slow.
    If your summary statistics are only a dict with a couple of numbers,
    the overhead should not be substantial.
    However, if your summary statistics are large numpy arrays
    or similar, this could cause overhead


    Parameters
    ----------

    n_procs: int, optional
        If set to None, the Number of cores is determined according to
        :func:`pyabc.sge.nr_cores_available`.
    """
    def __init__(self, n_procs=None):
        super().__init__()
        self._n_procs = n_procs

    @property
    def n_procs(self):
        if self._n_procs is not None:
            return self._n_procs
        return nr_cores_available()

    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        n_eval = Value(c_longlong)
        n_eval.value = 0

        n_particles = Value(c_longlong)
        n_particles.value = n

        queue = Queue()

        processes = [
            Process(target=work,
                    args=(sample_one, simulate_one, accept_one,
                          queue, n_eval, n_particles),
                    daemon=True)
            for _ in range(self.n_procs)
        ]

        for proc in processes:
            proc.start()

        id_results = []

        while len(id_results) < n:
            id_results.append(queue.get())

        for proc in processes:
            proc.join()

        # make sure all results are collected
        while not queue.empty():
            id_results.append(queue.get())

        # avoid bias toward short running evaluations
        id_results.sort(key=lambda x: x[0])
        id_results = id_results[:n]

        self.nr_evaluations_ = n_eval.value

        population = [res[1] for res in id_results]
        return population
