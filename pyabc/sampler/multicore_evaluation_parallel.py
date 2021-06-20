from multiprocessing import Process, Queue, Value
from ctypes import c_longlong
from .multicorebase import MultiCoreSampler
import numpy as np
import random
import cloudpickle as pickle
from jabbar import jabbar
from time import time
from .multicorebase import get_if_worker_healthy

DONE = "Done"


def work(simulate_one,
         queue,
         n_eval: Value,
         n_acc: Value,
         n: int,
         check_max_eval: bool,
         max_eval: int,
         all_accepted: bool,
         sample_factory):
    # unwrap arguments
    if isinstance(simulate_one, bytes):
        simulate_one = pickle.loads(simulate_one)

    random.seed()
    np.random.seed()

    sample = sample_factory()
    while (n_acc.value < n and
           (not all_accepted or n_eval.value < n) and
           (not check_max_eval or n_eval.value < max_eval)):
        with n_eval.get_lock():
            particle_id = n_eval.value
            n_eval.value += 1

        new_sim = simulate_one()
        sample.append(new_sim)

        if new_sim.accepted:

            # increase number of accepted particles
            with n_acc.get_lock():
                n_acc.value += 1

            # put into queue
            queue.put((particle_id, sample))

            # create empty sample and record until next accepted
            sample = sample_factory()

    # indicate worker finished
    queue.put(DONE)


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

    def sample_until_n_accepted(
            self, n, simulate_one, t, function_profile, *,
            max_eval=np.inf,
            all_accepted=False, ana_vars=None):
        s_def = time()
        n_eval = Value(c_longlong)
        n_eval.value = 0

        n_acc = Value(c_longlong)
        n_acc.value = 0

        queue = Queue()
        e_def = time()

        # wrap arguments
        s_wrap = time()

        if self.pickle:
            simulate_one = pickle.dumps(simulate_one)
        args = (simulate_one, queue,
                n_eval, n_acc, n,
                self.check_max_eval, max_eval, all_accepted,
                self._create_empty_sample)
        e_wrap = time()

        s_process = time()

        processes = [Process(target=work, args=args, daemon=self.daemon)
                     for _ in range(self.n_procs)]

        for proc in processes:
            proc.start()

        id_results = []

        # make sure all results are collected
        # and the queue is emptied to prevent deadlocks
        n_done = 0
        e_process = time()

        s_loop = time()

        with jabbar(total=n, enable=self.show_progress, keep=False) as bar:
            while n_done < len(processes):
                val = get_if_worker_healthy(processes, queue)
                if val == DONE:
                    n_done += 1
                else:
                    id_results.append(val)
                    bar.inc()
        e_loop = time()

        s_join = time()

        for proc in processes:
            proc.join()
        e_join = time()

        # avoid bias toward short running evaluations
        s_sort = time()

        id_results.sort(key=lambda x: x[0])
        id_results = id_results[:min(len(id_results), n)]
        e_sort = time()

        s_end = time()

        self.nr_evaluations_ = n_eval.value

        results = [res[1] for res in id_results]

        # create 1 to-be-returned sample from results
        sample = self._create_empty_sample()
        for result in results:
            sample += result

        if sample.n_accepted < n:
            sample.ok = False
        e_end = time()

        function_profile["eval_def"] += e_def - s_def
        function_profile["eval_warp"] += e_wrap - s_wrap
        function_profile["eval_process"] += e_process - s_process
        function_profile["eval_loop"] += e_loop - s_loop
        function_profile["eval_join"] += e_join - s_join
        function_profile["eval_sort"] += s_sort - e_sort
        function_profile["eval_end"] += s_end - e_end

        return sample, function_profile
