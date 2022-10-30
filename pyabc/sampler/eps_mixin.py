"""Client submission interface."""

from abc import ABC, abstractmethod
from time import sleep
from typing import Union

import cloudpickle as pickle
import numpy as np
from sortedcontainers import SortedList


class EPSMixin(ABC):
    """
    Provides sampling functionality for standard job submission clients.

    Mixin is the Python version of an interface.
    To be used in classes deriving from both `EPSMixin` and `Sampler`.

    Attributes
    ----------
    client:
        Client to submit jobs to. Provides a `submit()` function,
        which returns jobs, which provide `done()`, `cancel()` and `result()`
        functions.
    client_max_jobs:
        Maximum number of jobs that can submitted to the client at a time.
        If this value is smaller than the maximum number of cores provided by
        the distributed infrastructure, the infrastructure will not be utilized
        fully.
    default_pickle:
        Specify if the sampler uses python's default pickle function to
        communicate the submit function to python; if this is the case, a
        cloud-pickle based workaround is used to pickle the simulate and
        evaluate functions. This allows utilization of locally defined
        functions, which can not be pickled using default pickle, at the cost
        of an additional pickling overhead.
    batch_size:
        Number of parameter samples that are evaluated in one remote execution
        call. Batchsubmission can be used to reduce the communication overhead
        for fast (ms-s) model evaluations. Large batch sizes can result in un-
        necessary model evaluations. By default, batch_size=1, i.e. no
        batching is done.
    """

    def __init__(
        self,
        client,
        client_max_jobs: int,
        default_pickle: bool,
        batch_size: int,
    ):
        self.client = client
        self.client_max_jobs: int = client_max_jobs
        self.default_pickle: bool = default_pickle
        self.batch_size: int = batch_size

        self._simulate_accept_one: Union[bytes, None] = None

    @abstractmethod
    def client_cores(self) -> int:
        """Number of active client cores."""

    def _full_submit_function_pickle(self, job_id):
        """Default pickle function call wrapper."""
        # Unpickle function
        simulate_one = pickle.loads(self._simulate_accept_one)

        # Run batch_size evaluations and create list of tuples
        result_batch = []
        for j in range(self.batch_size):
            eval_result = simulate_one()
            eval_accept = eval_result.accepted
            result_batch.append((eval_result, eval_accept, job_id[j]))

        return result_batch

    def sample_until_n_accepted(
        self,
        n,
        simulate_one,
        t,
        *,
        max_eval=np.inf,
        all_accepted=False,
        ana_vars=None,
    ):
        # For default pickling
        if self.default_pickle:
            self._simulate_accept_one = pickle.dumps(simulate_one)
            full_submit_function = self._full_submit_function_pickle
        else:
            # For advanced pickling, e.g. cloudpickle
            def full_submit_function(job_id):
                # Run batch_size evaluations and create list of tuples
                result_batch = []
                for j in range(self.batch_size):
                    eval_result = simulate_one()
                    eval_accept = eval_result.accepted
                    result_batch.append((eval_result, eval_accept, job_id[j]))
                return result_batch

        # Run variables
        #  Counters for total and sequential (i.e. taking first-start into
        #  account) numbers of acceptance
        num_accepted = 0
        # Job identifier
        next_job_id = 0
        # List of running jobs
        running_jobs = []
        # List of results
        results = SortedList(key=lambda x: x[2])

        # Main loop, leave once we have enough material
        while True:
            # Gather results from finished jobs
            for job_id, job in running_jobs:
                if job.done():
                    batch = job.result()
                    results.update(batch)
                    num_accepted += sum(1 for ret in batch if ret[1])
                    running_jobs.remove((job_id, job))

            # Check whether all done
            if num_accepted >= n:
                # nth start index among accepted particles
                nth_accepted_id = [
                    result[2] for result in results if result[1]
                ][n - 1]
                # Cancel jobs started later than nth accepted one
                for job_id, job in running_jobs:
                    if job_id > nth_accepted_id:
                        running_jobs.remove((job_id, job))
                        job.cancel()
                # Break when no more jobs are running
                if len(running_jobs) == 0:
                    break

            # Submit jobs, only if:
            # * Number of jobs open < max_jobs
            # * Number of jobs open < self.scheduler_workers_running *
            #   worker_load_factor
            # * num_accepted_total < jobs required
            n_job_max = int(min(self.client_max_jobs, self.client_cores()))
            if (len(running_jobs) < n_job_max) and (num_accepted < n):
                n_job_req = n_job_max - len(running_jobs)
                for _ in range(0, n_job_req):
                    # Define job and batch ids
                    job_id = next_job_id
                    job_id_batch = [job_id + i for i in range(self.batch_size)]
                    next_job_id += self.batch_size
                    # Submit job
                    job = self.client.submit(
                        full_submit_function, job_id_batch
                    )
                    # Register job
                    running_jobs.append((job_id, job))

            # No need to be always awake
            sleep(0.01)

        # Create 1 to-be-returned sample from results
        sample = self._create_empty_sample()
        # Collect until n acceptances
        nth_accepted_id = [result[2] for result in results if result[1]][n - 1]
        while True:
            result = results.pop(0)
            sample.append(result[0])
            if result[2] == nth_accepted_id:
                break

        self.nr_evaluations_ = next_job_id

        return sample
