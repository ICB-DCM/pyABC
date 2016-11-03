from typing import List
from pyabc.parameters import Parameter
from abc import ABCMeta, abstractmethod


class MapWrapper():
    @abstractmethod
    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        pass


class MapWrapperDefault():
    """
        This is the basic map_wrapper implementation required for code compatibility reasons.
        There should be no need do initialize this on the user-level.
    """
    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        def map_fun_wrapper(_):
            while True:
                new_param = sample_one()
                new_sim = simulate_one(new_param)
                accepted = accept_one(new_sim)
                if accepted:
                    break
            return new_sim
        results = list(self.map_fun(map_fun_wrapper, [None] * n))
        return results

    def __init__(self, map_fun=map):
        self.map_fun = map_fun