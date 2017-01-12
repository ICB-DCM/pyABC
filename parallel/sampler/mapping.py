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

    map: the map function

    mapper_pickles: bool
        Whether the mapper handles the pickling itself
        or the MappingSampler class should handle serialization
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