"""Sample via dask."""

import numpy as np
from dask.distributed import Client

from .base import Sampler
from .eps_mixin import EPSMixin


class DaskDistributedSampler(EPSMixin, Sampler):
    """
    Parallelize with dask. This sampler is intended to be used with a
    pre-configured dask client, but is able to initialize client, scheduler and
    workers on its own on the local machine for testing/debugging purposes.

    Parameters
    ----------
    dask_client:
        The configured dask Client.
        If none is provided, then a local dask distributed cluster is created.
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
        of an additional pickling overhead. For dask, this workaround should
        not be necessary and it should be save to use default_pickle=false.
    batch_size:
        Number of parameter samples that are evaluated in one remote execution
        call. Batchsubmission can be used to reduce the communication overhead
        for fast (ms-s) model evaluations. Large batch sizes can result in un-
        necessary model evaluations. By default, batch_size=1, i.e. no
        batching is done.
    """

    def __init__(
        self,
        dask_client: Client = None,
        client_max_jobs: int = np.inf,
        default_pickle: bool = False,
        batch_size: int = 1,
    ):
        # Assign Client
        if dask_client is None:
            dask_client = Client()

        EPSMixin.__init__(
            self,
            client=dask_client,
            client_max_jobs=client_max_jobs,
            default_pickle=default_pickle,
            batch_size=batch_size,
        )
        Sampler.__init__(self)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['client']
        return d

    def client_cores(self) -> int:
        return sum(self.client.ncores().values())

    def shutdown(self):
        """Shutdown the dask client.
        If it was started without arguments, the
        local cluster that was started at the same time is also closed.
        """
        self.client.close()
