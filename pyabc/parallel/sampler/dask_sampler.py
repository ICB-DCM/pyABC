from.base import Sampler
import numpy as np
import time
from sortedcontainers import SortedListWithKey
from distributed import Client


class DaskDistributedSampler(Sampler):
    """
    Sample on a single core. No parallelization.
    """
    def __init__(self, dask_client=None, client_core_load_factor=1.2, client_max_jobs=200,  throttle_delay=0.0):
        self.nr_evaluations_ = 0

        # Assign Client
        if dask_client is None:
            dask_client = Client()
        self.my_client = dask_client

        # Client state
        self.client_cores = sum(self.my_client.ncores().values())

        # Client options
        self.throttle_delay = throttle_delay
        self.client_core_load_factor = client_core_load_factor
        self.client_max_jobs = client_max_jobs

        # Job state
        self.jobs_queued = 0

    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        num_accepted_total = 0
        next_job_id = 0
        running_jobs = []
        accepted_results = SortedListWithKey(key=lambda x: x[0])
        unprocessed_results = SortedListWithKey(key=lambda x: x[0])
        next_valid_index = -1

        def full_submit_function(param, job_id):
            eval_result = simulate_one(param)
            eval_accept = accept_one(eval_result)
            return eval_result, eval_accept, job_id

        # Main Loop, leave once we have enough material
        while True:
            # Gather finished jobs
            # make sure to track and update both total accepted and sequentially accepted jobs
            for curJob in running_jobs:
                if curJob.done():
                    remote_evaluated = curJob.result()
                    running_jobs.remove(curJob)
                    remote_result = remote_evaluated[0]
                    remote_accept = remote_evaluated[1]
                    remote_jobid = remote_evaluated[2]
                    # print("Received result on job ", remote_jobid)
                    unprocessed_results.add((remote_jobid, remote_accept, remote_result))
                    if remote_accept:
                        num_accepted_total += 1

            if len(unprocessed_results) > 0:
                next_index = unprocessed_results[0][0]
            else:
                next_index = np.nan
            while next_index == next_valid_index+1:
                seq_evaluated = unprocessed_results.pop(0)
                if seq_evaluated[1]:
                    accepted_results.add((seq_evaluated[0], seq_evaluated[2]))
                next_valid_index += 1
                if len(unprocessed_results) > 0:
                    next_index = unprocessed_results[0][0]
                else:
                    next_index = np.nan

            # If num_accepted_sequential >= n
            # return the first n accepted results
            num_accepted_sequential = len(accepted_results)
            if num_accepted_sequential >= n:
                break

            # Update informations on scheduler state
            self.client_cores = sum(self.my_client.ncores().values())
            # Only submit more jobs if:
            # Number of jobs open < max_jobs
            # Number of jobs open < self.scheduler_workers_running * worker_load_factor
            # num_accepted_total < jobs required
            if (len(running_jobs) < self.client_max_jobs) and \
                    (len(running_jobs) < self.client_cores * self.client_core_load_factor) and \
                    (num_accepted_total < n):
                for _ in range(0, np.minimum(self.client_max_jobs,
                                             self.client_cores * self.client_core_load_factor).astype(int) - len(running_jobs)):
                    new_param = sample_one()
                    running_jobs.append(self.my_client.submit(full_submit_function, new_param, next_job_id))
                    # print("Submitted job ", next_job_id)
                    next_job_id += 1
            # Wait for scheduler_throttle_delay seconds
            time.sleep(self.throttle_delay)

        # Cancel all unfinished jobs
        for curJob in running_jobs:
            curJob.cancel()
        returned_results = []
        while len(returned_results) < n:
            cur_res = accepted_results.pop(0)
            returned_results.append(cur_res[1])
            self.nr_evaluations_ = cur_res[0]
        return returned_results


