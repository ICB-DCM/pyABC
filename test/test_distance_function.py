import numpy as np
import scipy as sp
from pyabc import (PercentileDistance,
                   MinMaxDistance,
                   PNormDistance,
                   AdaptivePNormDistance,
                   AggregatedDistance,
                   AdaptiveAggregatedDistance)


from pyabc.distance import (
    median_absolute_deviation,
    mean_absolute_deviation,
    standard_deviation,
    bias,
    root_mean_square_deviation,
    median_absolute_deviation_to_observation,
    mean_absolute_deviation_to_observation,
    combined_median_absolute_deviation,
    combined_mean_absolute_deviation,
    standard_deviation_to_observation,
    span,
    mean,
    median)


class MockABC:
    def __init__(self, samples):
        self.samples = samples

    def sample_from_prior(self):
        return self.samples


def test_single_parameter():
    dist_f = MinMaxDistance(measures_to_use=["a"])
    abc = MockABC([{"a": -3}, {"a": 3}, {"a": 10}])
    dist_f.initialize(0, abc.sample_from_prior)
    d = dist_f({"a": 1}, {"a": 2})
    assert 1 / 13 == d


def test_two_parameters_but_only_one_used():
    dist_f = MinMaxDistance(measures_to_use=["a"])
    abc = MockABC([{"a": -3, "b": 2}, {"a": 3, "b": 3}, {"a": 10, "b": 4}])
    dist_f.initialize(0, abc.sample_from_prior)
    d = dist_f({"a": 1, "b": 10}, {"a": 2, "b": 12})
    assert 1 / 13 == d


def test_two_parameters_and_two_used():
    dist_f = MinMaxDistance(measures_to_use=["a", "b"])
    abc = MockABC([{"a": -3, "b": 2}, {"a": 3, "b": 3}, {"a": 10, "b": 4}])
    dist_f.initialize(0, abc.sample_from_prior)
    d = dist_f({"a": 1, "b": 10}, {"a": 2, "b": 12})
    assert 1 / 13 + 2 / 2 == d


def test_single_parameter_percentile():
    dist_f = PercentileDistance(measures_to_use=["a"])
    abc = MockABC([{"a": -3}, {"a": 3}, {"a": 10}])
    dist_f.initialize(0, abc.sample_from_prior)
    d = dist_f({"a": 1}, {"a": 2})
    expected = (
        1 / (sp.percentile([-3, 3, 10], 80) - sp.percentile([-3, 3, 10], 20))
    )
    assert expected == d


def test_pnormdistance():
    abc = MockABC([{'s1': -1, 's2': -1, 's3': -1},
                   {'s1': -1, 's2': 0, 's3': 1}])
    x_0 = {'s1': 0, 's2': 0, 's3': 1}

    # first test that for PNormDistance, the weights stay constant
    dist_f = PNormDistance()
    dist_f.initialize(0, abc.sample_from_prior, x_0=x_0)

    # call distance function, also to initialize w
    d = dist_f(abc.sample_from_prior()[0], abc.sample_from_prior()[1], t=0)
    expected = pow(1**2 + 2**2, 1 / 2)
    assert expected == d

    assert sum(abs(a - b) for a, b in
               zip(list(dist_f.weights[0].values()), [1, 1, 1])) < 0.01


def test_adaptivepnormdistance():
    """
    Only tests basic running.
    """
    abc = MockABC([{'s1': -1, 's2': -1, 's3': -1},
                   {'s1': -1, 's2': 0, 's3': 1}])
    x_0 = {'s1': 0, 's2': 0, 's3': 1}

    scale_functions = [
        median_absolute_deviation,
        mean_absolute_deviation,
        standard_deviation,
        bias,
        root_mean_square_deviation,
        median_absolute_deviation_to_observation,
        mean_absolute_deviation_to_observation,
        combined_median_absolute_deviation,
        combined_mean_absolute_deviation,
        standard_deviation_to_observation
    ]

    for scale_function in scale_functions:
        dist_f = AdaptivePNormDistance(
            scale_function=scale_function)
        dist_f.initialize(0, abc.sample_from_prior, x_0=x_0)
        dist_f(abc.sample_from_prior()[0], abc.sample_from_prior()[1], t=0)

    # test max weight ratio
    for scale_function in scale_functions:
        dist_f = AdaptivePNormDistance(
            scale_function=scale_function,
            max_weight_ratio=20)
        dist_f.initialize(0, abc.sample_from_prior, x_0=x_0)
        dist_f(abc.sample_from_prior()[0], abc.sample_from_prior()[1], t=0)


def test_aggregateddistance():
    abc = MockABC([{'s0': -1, 's1': -1},
                   {'s0': -1, 's1': 0}])
    x_0 = {'s0': 0, 's1': 0}

    def distance0(x, x_0):
        return abs(x['s0'] - x_0['s0'])

    def distance1(x, x_0):
        return np.sqrt((x['s1'] - x_0['s1'])**2)

    distance = AggregatedDistance(
        [distance0, distance1])
    distance.initialize(0, abc.sample_from_prior, x_0=x_0)
    val = distance(
        abc.sample_from_prior()[0], abc.sample_from_prior()[1], t=0)
    assert isinstance(val, float)


def test_adaptiveaggregateddistance():
    abc = MockABC([{'s0': -1, 's1': -1},
                   {'s0': -1, 's1': 0}])
    x_0 = {'s0': 0, 's1': 0}

    def distance0(x, x_0):
        return abs(x['s0'] - x_0['s0'])

    def distance1(x, x_0):
        return np.sqrt((x['s1'] - x_0['s1'])**2)

    scale_functions = [span, mean, median]
    for scale_function in scale_functions:
        distance = AdaptiveAggregatedDistance(
            [distance0, distance1], scale_function=scale_function)
        distance.initialize(0, abc.sample_from_prior, x_0=x_0)
        val = distance(
            abc.sample_from_prior()[0], abc.sample_from_prior()[1], t=0)
        assert isinstance(val, float)
