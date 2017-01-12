from abc import ABC, abstractmethod


class Sampler(ABC):
    def __init__(self):
        self.nr_evaluations_ = 0

    @abstractmethod
    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        pass