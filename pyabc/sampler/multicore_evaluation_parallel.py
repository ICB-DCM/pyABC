from multiprocessing import Process, Queue, Value
from ctypes import c_longlong
from .base import Sampler
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


class MulticoreEvalParallelSampler(Sampler):
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
    """
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
            for _ in range(nr_cores_available())
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
