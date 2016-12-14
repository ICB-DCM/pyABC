from abc import ABC, abstractmethod
import dill as pickle
import functools


class Sampler(ABC):
    @abstractmethod
    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        pass


class SingleCoreSampler(Sampler):
    def __init__(self):
        self.nr_evaluations_ = 0

    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        nr_simulations = 0
        results = []
        for _ in range(n):
            while True:
                new_param = sample_one()
                new_sim = simulate_one(new_param)
                nr_simulations += 1
                if accept_one(new_sim):
                    break
            results.append(new_sim)
        self.nr_evaluations_ = nr_simulations
        assert len(results) == n
        return results


class MappingSampler(Sampler):
    """
    This is the basic sampler implementation required for code compatibility reasons.
    There should be no need do initialize this on the user-level.
    """
    def __init__(self, map=map):
        self.map = map
        self.nr_evaluations_ = 0

    @staticmethod
    def map_function(str_sample_pickle, str_simulate_pickle, str_accept_pickle, _):
        import numpy as np
        import random

        sample_from_pickle = pickle.loads(str_sample_pickle)
        simulate_from_pickle = pickle.loads(str_simulate_pickle)
        accept_from_pickle = pickle.loads(str_accept_pickle)

        np.random.seed()
        random.seed()
        nr_simulations = 0
        while True:
            new_param = sample_from_pickle()
            new_sim = simulate_from_pickle(new_param)
            nr_simulations += 1
            if accept_from_pickle(new_sim):
                break
        return new_sim, nr_simulations

    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        sample_pickle = pickle.dumps(sample_one)
        simulate_pickle = pickle.dumps(simulate_one)
        accept_pickle = pickle.dumps(accept_one)

        counted_results = list(self.map(functools.partial(self.map_function, sample_pickle,
                                                          simulate_pickle, accept_pickle), [None] * n))
        self.nr_evaluations_ = sum(nr for res, nr in counted_results)
        results = [res for res, nr in counted_results]
        return results
