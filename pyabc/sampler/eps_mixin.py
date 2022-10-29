from abc import ABC, abstractmethod
from typing import Union

import cloudpickle as pickle
import numpy as np
from sortedcontainers import SortedListWithKey


class EPSMixin(ABC):
    """


    Mixin is the Python version of an interface.

    Attributes
    ----------
    client:
        Client to perform the sampling on.
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
        num_accepted_total = num_accepted_sequential = 0
        # Job identifier
        next_job_id = 0
        #  List of running jobs
        running_jobs = []
        unprocessed_results = SortedListWithKey(key=lambda x: x[0])
        all_results = SortedListWithKey(key=lambda x: x[0])
        next_valid_index = -1

        # Main loop, leave once we have enough material
        while True:
            # Gather finished jobs
            # make sure to track and update both
            # total accepted and sequentially
            # accepted jobs
            for cur_job in running_jobs:
                if cur_job.done():
                    remote_batch = cur_job.result()
                    running_jobs.remove(cur_job)
                    # Extract particles
                    for i in range(self.batch_size):
                        remote_evaluated = remote_batch[i]
                        (
                            remote_result,
                            remote_accept,
                            remote_jobid,
                        ) = remote_evaluated
                        # print("Received result on job ", remote_jobid)
                        unprocessed_results.add(
                            (remote_jobid, remote_accept, remote_result)
                        )
                        if remote_accept:
                            num_accepted_total += 1

            next_index = (
                unprocessed_results[0][0]
                if len(unprocessed_results) > 0
                else np.nan
            )

            # Process results
            while next_index == next_valid_index + 1:
                seq_jobid, seq_accept, seq_result = unprocessed_results.pop(0)
                # add to all_results
                all_results.add((seq_jobid, seq_result))
                # update accepted counter
                if seq_accept:
                    num_accepted_sequential += 1
                next_valid_index += 1
                next_index = (
                    unprocessed_results[0][0]
                    if len(unprocessed_results) > 0
                    else np.nan
                )

            # If num_accepted >= n
            # return the first n accepted results
            if num_accepted_sequential >= n:
                break

            # Update information on scheduler state
            # Only submit more jobs if:
            # * Number of jobs open < max_jobs
            # * Number of jobs open < self.scheduler_workers_running *
            #   worker_load_factor
            # * num_accepted_total < jobs required
            if (
                (len(running_jobs) < self.client_max_jobs)
                and (len(running_jobs) < self.client_cores())
                and (num_accepted_total < n)
            ):
                for _ in range(
                    0,
                    np.minimum(
                        self.client_max_jobs, self.client_cores()
                    ).astype(int)
                    - len(running_jobs),
                ):
                    job_id_batch = []
                    for _ in range(self.batch_size):
                        job_id_batch.append(next_job_id)
                        next_job_id += 1

                    running_jobs.append(
                        self.client.submit(full_submit_function, job_id_batch)
                    )

        # cancel all unfinished jobs
        for cur_job in running_jobs:
            cur_job.cancel()

        # create 1 to-be-returned sample from all results
        sample = self._create_empty_sample()
        counter_accepted = 0
        self.nr_evaluations_ = 0
        while counter_accepted < n:
            cur_res = all_results.pop(0)
            particle = cur_res[1]
            sample.append(particle)
            if particle.accepted:
                counter_accepted += 1
            # n_eval is latest job_id + 1
            self.nr_evaluations_ = max(self.nr_evaluations_, cur_res[0] + 1)

        return sample
