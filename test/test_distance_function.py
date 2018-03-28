import scipy as sp
from pyabc import (PercentileDistanceFunction,
                   MinMaxDistanceFunction,
                   PNormDistance,
                   WeightedPNormDistance)


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
    assert 1/13 + 2/2 == d


def test_single_parameter_percentile():
    dist_f = PercentileDistanceFunction(measures_to_use=["a"])
    abc = MockABC([{"a": -3}, {"a": 3}, {"a": 10}])
    dist_f.initialize(abc.sample_from_prior())
    d = dist_f({"a": 1}, {"a": 2})
    expected = (
        1 / (sp.percentile([-3, 3, 10], 80) - sp.percentile([-3, 3, 10], 20))
    )
    assert expected == d


def test_pnormdistance():
    abc = MockABC([{'s1': -1, 's2': -1, 's3': -1},
                   {'s1': -1, 's2': 0, 's3': 1}])

    # fist test that for PNormDistance, the weights stay constant
    dist_f = PNormDistance(p=2)
    dist_f.initialize(abc.sample_from_prior())
    assert sum(abs(a-b) for a, b in
               zip(list(dist_f.w.values()), [1, 1, 1])) < 0.01


def test_weightedpnormdistance():
    abc = MockABC([{'s1': -1, 's2': -1, 's3': -1},
                   {'s1': -1, 's2': 0, 's3': 1}])

    # now test that the weights adapt correctly for a weighted distance
    scale_type = WeightedPNormDistance.SCALE_TYPE_MAD
    dist_f = WeightedPNormDistance(p=2,
                                   adaptive=True,
                                   scale_type=scale_type)
    dist_f.initialize(abc.sample_from_prior())
    # the weights are adapted using MAD, and then divided by the mean
    # here mean = 4/3, so (noting that in addition the MAD of 0 in s1 is set to
    # 1) we expect the following
    print(dist_f.w)
    assert sum(abs(a-b) for a, b in
               zip(list(dist_f.w.values()), [0.75, 1.5, 0.75])) < 0.01
