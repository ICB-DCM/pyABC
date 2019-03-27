from distributed import Client
from .base import Sampler
from .eps_mixin import EPSMixin
import numpy as np


class DaskDistributedSampler(EPSMixin, Sampler):
    """
    Parallelize with dask. This sampler is intended to be used with a
    pre-configured dask client, but is able to initialize client, scheduler and
    workers on its own on the local machine for testing/debugging purposes.

    Parameters
    ----------

    dask_client: dask.Client, optional
        The configured dask Client.
        If none is provided, then a local dask distributed cluster is created.

    client_max_jobs:
        Maximum number of jobs that can submitted to the client at a time.
        If this value is smaller than the maximum number of cores provided by
        the distributed infrastructure, the infrastructure will not be utilized
        fully.

    default_pickle:
        Specify if the sampler uses pythons default pickle function to
        communicate the submit function to python; if this is the case, a
        cloud-pickle based workaround is used to pickle the simulate and
        evaluate functions. This allows utilization of locally defined
        functions, which can not be pickled using default pickle, at the cost
        of an additional pickling overhead. For dask, this workaround should
        not be necessary and it should be save to use default_pickle=false.

    batch_size: int, optional
        Number of parameter samples that are evaluated in one remote execution
        call. Batchsubmission can be used to reduce the communication overhead
        for fast (ms-s) model evaluations. Large batch sizes can result in un-
        necessary model evaluations. By default, batch_size=1, i.e. no
        batching is done.

    """

    def __init__(self, dask_client=None, client_max_jobs=np.inf,
                 default_pickle=False, batch_size=1):
        super().__init__()

        # Assign Client
        if dask_client is None:
            dask_client = Client()
        self.my_client = dask_client

        # Client options
        self.client_max_jobs = client_max_jobs

        # Job state
        self.jobs_queued = 0

        # For dask, we use cloudpickle by default
        self.default_pickle = default_pickle

        # Batchsize
        self.batch_size = batch_size

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['my_client']
        return d

    def client_cores(self):
        return sum(self.my_client.ncores().values())
