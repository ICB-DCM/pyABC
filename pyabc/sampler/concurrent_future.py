from .base import Sampler
from .eps_mixin import EPSMixin


class ConcurrentFutureSampler(EPSMixin, Sampler):
    """
    Parallelize with an arbitrary sampler that implements the python concurrent
    futures executor interface. Specifically, it needs to implement a "submit"
    function that is able to evaluate arbitrary function handles and return a
    concurrent future result object

    Parameters
    ----------

    cfuture_executor: concurrent.futures.Executor, required
        Configured object that implements the concurrent.futures.Executor
        interface

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
        of an additional pickling overhead.

    batchsize: int, optional
        Number of parameter samples that are evaluated in one remote execution
        call. Batchsubmission can be used to reduce the communication overhead
        for fast (ms-s) model evaluations. Large batchsizes can result in un-
        necessary model evaluations. By default, batchsize=1, i.e. no batching
        is done


    """

    def __init__(self, cfuture_executor=None, client_max_jobs=200,
                 default_pickle=True, batchsize=1):
        super().__init__()

        # Assign Client
        self.my_client = cfuture_executor

        # Client options
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
