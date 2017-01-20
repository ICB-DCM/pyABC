import functools
import random

import dill as pickle
import numpy as np

from .base import Sampler


class MappingSampler(Sampler):
    """
    Parallelize via a map operation

    Parameters
    ----------

    map: A function which works like the built in `map`.

        The map can be really any generic map operations. Possible canidates
        include

        * multiprocessing.pool.map
          (see https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool)
        * :class:`pyabc.parallel.sge.SGE`'s map method. This mapper is useful
          in SGE-like environments where you don't want to start workers which
          run forever.
        * Dask's distributed `distributed.Client`'s map
          (see https://distributed.readthedocs.io/en/latest/api.html#client)
        * IPython parallel' map
          (see http://ipyparallel.readthedocs.io/en/latest/task.html#quick-and-easy-parallelism)

        and many other implementations.

        Each of the mapped function calls samples until it gets one accepted
        particle. This could have a performance impact if one of the sample
        tasks runs very long and all the other tasks are already finished.
        The sampler then has to wait until the last sample task is finished.

    mapper_pickles: bool, optional
        Whether the mapper handles the pickling itself
        or the MappingSampler class should handle serialization.

        The default is `False`. However, this might have a substantial
        performance impact as much serialization and deserialization is done,
        which could limit overall performace if the model evaluations are
        comparatively fast. For example, for the :class:`pyabc.parallel.sge.SGE`
        mapper, this option should be set to `True` for better performance.
    """
    def __init__(self, map=map, mapper_pickles=False):
        super().__init__()
        self.map = map
        self.pickle, self.unpickle = (identity, identity) if mapper_pickles else (pickle.dumps, pickle.loads)

    def __getstate__(self):
        return self.pickle, self.unpickle, self.nr_evaluations_

    def __setstate__(self, state):
        self.pickle, self.unpickle, self.nr_evaluations_ = state

    def map_function(self, sample_pickle, simulate_pickle, accept_pickle, _):
        sample_one = self.unpickle(sample_pickle)
        simulate_one = self.unpickle(simulate_pickle)
        accept_one = self.unpickle(accept_pickle)

        np.random.seed()
        random.seed()
        nr_simulations = 0
        while True:
            new_param = sample_one()
            new_sim = simulate_one(new_param)
            nr_simulations += 1
            if accept_one(new_sim):
                break
        return new_sim, nr_simulations

    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        sample_pickle = self.pickle(sample_one)
        simulate_pickle = self.pickle(simulate_one)
        accept_pickle = self.pickle(accept_one)

        map_function = functools.partial(self.map_function, sample_pickle, simulate_pickle, accept_pickle)

        counted_results = list(self.map(map_function, [None] * n))
        self.nr_evaluations_ = sum(nr for res, nr in counted_results)
        results = [res for res, nr in counted_results]
        return results


def identity(x):
    return x