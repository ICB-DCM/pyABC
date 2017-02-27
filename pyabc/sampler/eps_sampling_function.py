import numpy as np
import cloudpickle as pickle
from sortedcontainers import SortedListWithKey


def full_submit_function_pickle(self, param, job_id):
    simulate_one, accept_one = pickle.loads(self.simulate_accept_one)
    result_batch = []
    for j in range(self.batchsize):
        eval_result = simulate_one(param[j])
        eval_accept = accept_one(eval_result)
        result_batch.append((eval_result, eval_accept, job_id[j]))
    return result_batch


def sample_until_n_accepted_proto(self, sample_one, simulate_one, accept_one,
                                  n):
    # For default pickling
    if self.default_pickle:
        self.simulate_accept_one = pickle.dumps((simulate_one, accept_one))
        full_submit_function = self.full_submit_function_pickle
    else:
        # For advanced pickling, e.g. cloudpickle
        def full_submit_function(param, job_id):
            result_batch = []
            for j in range(self.batchsize):
                eval_result = simulate_one(param[j])
                eval_accept = accept_one(eval_result)
                result_batch.append((eval_result, eval_accept, job_id[j]))
            return result_batch

    num_accepted_total = 0
    next_job_id = 0
    running_jobs = []
    accepted_results = SortedListWithKey(key=lambda x: x[0])
    unprocessed_results = SortedListWithKey(key=lambda x: x[0])
    next_valid_index = -1

    # Main Loop, leave once we have enough material
    while True:
        # Gather finished jobs
        # make sure to track and update both total accepted and sequentially
        # accepted jobs
        for curJob in running_jobs:
            if curJob.done():
                remote_batch = curJob.result()
                running_jobs.remove(curJob)
                for i in range(self.batchsize):
                    remote_evaluated = remote_batch[i]
                    remote_result = remote_evaluated[0]
                    remote_accept = remote_evaluated[1]
                    remote_jobid = remote_evaluated[2]
                    # print("Received result on job ", remote_jobid)
                    unprocessed_results.add((remote_jobid, remote_accept,
                                             remote_result))
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
        # Only submit more jobs if:
        # Number of jobs open < max_jobs
        # Number of jobs open < self.scheduler_workers_running *
        # worker_load_factor
        # num_accepted_total < jobs required
        if (len(running_jobs) < self.client_max_jobs) and \
                (len(running_jobs) < self.client_cores()) and \
                (num_accepted_total < n):
            for _ in range(0,
                           np.minimum(self.client_max_jobs,
                                      self.client_cores()).astype(int)
                           - len(running_jobs)):
                para_batch = []
                job_id_batch = []
                for i in range(self.batchsize):
                    para_batch.append(sample_one())
                    job_id_batch.append(next_job_id)
                    next_job_id += 1

                running_jobs.append(self.my_client.submit(full_submit_function,
                                                          para_batch,
                                                          job_id_batch))

    # Cancel all unfinished jobs
    for curJob in running_jobs:
        curJob.cancel()
    returned_results = []
    while len(returned_results) < n:
        cur_res = accepted_results.pop(0)
        returned_results.append(cur_res[1])
        self.nr_evaluations_ = cur_res[0]
    return returned_results
