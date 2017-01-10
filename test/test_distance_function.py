import scipy as sp
from pyabc import PercentileDistanceFunction, MinMaxDistanceFunction


class MockABC:
    def __init__(self, samples):
        self.samples = samples

    def sample_from_prior(self):
        return self.samples


def test_single_parameter():
    dist_f = MinMaxDistanceFunction(measures_to_use=["a"])
    abc = MockABC([{"a": -3}, {"a": 3}, {"a": 10}])
    dist_f.initialize(abc.sample_from_prior())
    d = dist_f({"a": 1}, {"a": 2})
    assert 1/13 == d

def test_two_parameters_but_only_one_used():
    dist_f = MinMaxDistanceFunction(measures_to_use=["a"])
    abc = MockABC([{"a": -3, "b": 2}, {"a": 3, "b": 3}, {"a": 10, "b": 4}])
    dist_f.initialize(abc.sample_from_prior())
    d = dist_f({"a": 1, "b": 10}, {"a": 2, "b": 12})
    assert 1/13 == d

def test_two_parameters_and_two_used():
    dist_f = MinMaxDistanceFunction(measures_to_use=["a", "b"])
    abc = MockABC([{"a": -3, "b": 2}, {"a": 3, "b": 3}, {"a": 10, "b": 4}])
    dist_f.initialize(abc.sample_from_prior())
    d = dist_f({"a": 1, "b": 10}, {"a": 2, "b": 12})
    assert 1/13 +2/2 == d

def test_single_parameter_percentile():
    dist_f = PercentileDistanceFunction(measures_to_use=["a"])
    abc = MockABC([{"a": -3}, {"a": 3}, {"a": 10}])
    dist_f.initialize(abc.sample_from_prior())
    d = dist_f({"a": 1}, {"a": 2})
    assert 1/(sp.percentile([-3,3,10], 80)-sp.percentile([-3,3,10], 20)) == d

