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

    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        def map_function(_):

            import numpy as np
            import random
            np.random.seed()
            random.seed()

            while True:
                new_param = sample_one()
                new_sim = simulate_one(new_param)
                accepted = accept_one(new_sim)
                if accepted:
                    break
            return new_sim

        results = list(self.map(map_function, [None] * n))
        return results
