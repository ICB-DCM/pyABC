from distributed import Client
from .base import Sampler
from .eps_sampling_function import sample_until_n_accepted_proto, full_submit_function_pickle


class DaskDistributedSampler(Sampler):
    """
    Parallelize with dask. This sampler requires a pre-configured dask cluster

    Parameters
    ----------

    dask_client: dask.Client, optional
        The configured dask Client.
        If none is provided, then a local dask distributed Cluster is created.
    """
    sample_until_n_accepted = sample_until_n_accepted_proto
    full_submit_function_pickle = full_submit_function_pickle

    def __init__(self, dask_client=None, client_core_load_factor=1.2,
                 client_max_jobs=200,  throttle_delay=0.0,
                 default_pickle=False, batchsize=1):
        self.nr_evaluations_ = 0

        # Assign Client
        if dask_client is None:
            dask_client = Client()
        self.my_client = dask_client

        # Client options
        self.throttle_delay = throttle_delay
        self.client_core_load_factor = client_core_load_factor
        self.client_max_jobs = client_max_jobs

        # Job state
        self.jobs_queued = 0

        # For dask, we use cloudpickle by default
        self.default_pickle = default_pickle

        # Batchsize
        self.batchsize = batchsize

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['my_client']
        return d

    def client_cores(self):
        return sum(self.my_client.ncores().values())

