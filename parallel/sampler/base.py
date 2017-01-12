from abc import ABC, abstractmethod


class Sampler(ABC):
    @abstractmethod
    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        pass