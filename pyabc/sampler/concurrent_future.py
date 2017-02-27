from .base import Sampler
from .eps_sampling_function import sample_until_n_accepted_proto, full_submit_function_pickle


class ConcurrentFutureSampler(Sampler):
    """
    Sample on a single core. No parallelization.
    """
    sample_until_n_accepted = sample_until_n_accepted_proto
    full_submit_function_pickle = full_submit_function_pickle

    def __init__(self, cfuture_executor=None, client_core_load_factor=1.2, client_max_jobs=200,  throttle_delay=0.0,
                 default_pickle=True, batchsize=1):
        self.nr_evaluations_ = 0

        # Assign Client
        self.my_client = cfuture_executor

        # Client options
        self.throttle_delay = throttle_delay
        self.client_core_load_factor = client_core_load_factor
        self.client_max_jobs = client_max_jobs

        # Job state
        self.jobs_queued = 0

        # Empty functions
        self.simulate_one = None
        self.accept_one = None

        # Option pickling
        self.default_pickle = default_pickle

        # Batchsize
        self.batchsize = batchsize

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['my_client']
        return d

    def client_cores(self):
        return self.client_max_jobs



