from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        pass


class MappingSampler(Sampler):
    """
    This is the basic sampler implementation required for code compatibility reasons.
    There should be no need do initialize this on the user-level.
    """
    def __init__(self, map=map):
        self.map = map
        self.nr_evaluations_ = 0

    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        def map_function(_):

            import numpy as np
            import random
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

        counted_results = list(self.map(map_function, [None] * n))
        self.nr_evaluations_ = sum(nr for res, nr in counted_results)
        results = [res for res, nr in counted_results]
        return results
