import functools
import random

import dill as pickle
import numpy as np

from .base import Sampler


class MappingSampler(Sampler):
    """
    Parallelize via a map operation.
    This sampler can be applied in a multi-core or in a distributed
    setting.

    Parameters
    ----------

    map: map like function

        A function which works like the built in `map`.
        The map can be really any generic map operations. Possible candidates
        include:

        * multiprocessing.pool.map
          (see https://docs.python.org/3/library/\
multiprocessing.html#multiprocessing.pool.Pool)
        * :class:`pyabc.sge.SGE`'s map method. This mapper is useful
          in SGE-like environments where you don't want to start workers which
          run forever.
        * Dask's distributed `distributed.Client`'s map
          (see https://distributed.readthedocs.io/en/latest/api.html#client)
        * IPython parallel' map (see http://ipyparallel.readthedocs.io/en/\
latest/task.html#quick-and-easy-parallelism)

        and many other implementations.

        Each of the mapped function calls samples until it gets one accepted
        particle. This could have a performance impact if one of the sample
        tasks runs very long and all the other tasks are already finished.
        The sampler then has to wait until the last sample task is finished.

    mapper_pickles: bool, optional
        Whether the mapper handles the pickling itself
        or the MappingSampler class should handle serialization.

        The default is `False`.
        While this setting is compatible with a larger range of
        map functions, its performance can be suboptimal.
        As possibly too much serialization and deserialization is done,
        which could limit overall performace if the model evaluations are
        comparatively fast.
        The passed map function might implement more efficient serialization.
        For example, for the
        :class:`pyabc.sge.SGE` mapper, this option should be set to
        `True` for better performance.
    """

    def __init__(self, map_=map, mapper_pickles: bool = False):
        super().__init__()
        self.map_ = map_
        self.pickle, self.unpickle = ((identity, identity)
                                      if mapper_pickles
                                      else (pickle.dumps, pickle.loads))

    def __getstate__(self):
        return (self.pickle, self.unpickle,
                self.nr_evaluations_, self.sample_factory)

    def __setstate__(self, state):
        (self.pickle, self.unpickle, self.nr_evaluations_,
         self.sample_factory) = state

    def map_function(self, simulate_one, _):
        simulate_one = self.unpickle(simulate_one)

        np.random.seed()
        random.seed()
        nr_simulations = 0
        sample = self._create_empty_sample()

        while True:
            new_sim = simulate_one()
            nr_simulations += 1
            sample.append(new_sim)
            if new_sim.accepted:
                break

        return sample, nr_simulations

    def sample_until_n_accepted(
            self, n, simulate_one, max_eval=np.inf, all_accepted=False):
        # pickle them as a tuple instead of individual pickling
        # this should save time and should make better use of
        # shared references.
        # Correct usage of shared references might even be necessary
        # to ensure correct working, depending on the details of the
        # model implementations.
        sample_simulate_accept = self.pickle(simulate_one)
        map_function = functools.partial(self.map_function,
                                         sample_simulate_accept)

        counted_results = list(self.map_(map_function, [None] * n))
        counted_results = filter(lambda x: not isinstance(x, Exception),
                                 counted_results)
        results, evals = zip(*counted_results)

        # count all evaluations
        self.nr_evaluations_ = sum(evals)

        # aggregate all results to 1 to-be-returned sample
        sample = self._create_empty_sample()
        for result in results:
            sample += result

        return sample


def identity(x):
    return x
