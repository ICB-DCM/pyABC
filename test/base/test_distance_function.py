import numpy as np
import scipy as sp
import scipy.stats
import tempfile

from pyabc.distance import (
    PercentileDistance,
    MinMaxDistance,
    PNormDistance,
    AdaptivePNormDistance,
    AggregatedDistance,
    AdaptiveAggregatedDistance,
    NormalKernel,
    IndependentNormalKernel,
    IndependentLaplaceKernel,
    BinomialKernel,
    PoissonKernel,
    NegativeBinomialKernel,
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
    median,
    SCALE_LIN,
)
from pyabc.storage import load_dict_from_json


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
        1 / (np.percentile([-3, 3, 10], 80) - np.percentile([-3, 3, 10], 20))
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

    expected = pow(1**2 + 2**2, 1/2)
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
        standard_deviation_to_observation,
    ]

    for scale_function in scale_functions:
        dist_f = AdaptivePNormDistance(
            scale_function=scale_function)
        dist_f.initialize(0, abc.sample_from_prior, x_0=x_0)
        dist_f(abc.sample_from_prior()[0], abc.sample_from_prior()[1], t=0)
        assert dist_f.weights[0] != {'s1': 1, 's2': 1, 's3': 1}

    # test max weight ratio
    for scale_function in scale_functions:
        dist_f = AdaptivePNormDistance(
            scale_function=scale_function,
            max_weight_ratio=20)
        dist_f.initialize(0, abc.sample_from_prior, x_0=x_0)
        dist_f(abc.sample_from_prior()[0], abc.sample_from_prior()[1], t=0)
        assert dist_f.weights[0] != {'s1': 1, 's2': 1, 's3': 1}


def test_adaptivepnormdistance_initial_weights():
    abc = MockABC([{'s1': -1, 's2': -1, 's3': -1},
                   {'s1': -1, 's2': 0, 's3': 1}])
    x_0 = {'s1': 0, 's2': 0, 's3': 1}

    # first test that for PNormDistance, the weights stay constant
    initial_weights = {'s1': 1, 's2': 2, 's3': 3}
    dist_f = AdaptivePNormDistance(initial_weights=initial_weights)
    dist_f.initialize(0, abc.sample_from_prior, x_0=x_0)
    assert dist_f.weights[0] == initial_weights

    # call distance function
    d = dist_f(abc.sample_from_prior()[0], abc.sample_from_prior()[1], t=0)
    expected = pow(sum([(2*1)**2, (3*2)**2]), 1/2)
    assert expected == d

    # check updating works
    dist_f.update(1, abc.sample_from_prior)
    assert dist_f.weights[1] != dist_f.weights[0]


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
        assert (distance.weights[0] != [1, 1]).any()


def test_adaptiveaggregateddistance_calibration():
    abc = MockABC([{'s0': -1, 's1': -1},
                   {'s0': -1, 's1': 0}])
    x_0 = {'s0': 0, 's1': 0}

    def distance0(x, x_0):
        return abs(x['s0'] - x_0['s0'])

    def distance1(x, x_0):
        return np.sqrt((x['s1'] - x_0['s1'])**2)

    scale_functions = [span, mean, median]
    initial_weights = np.array([2, 3])
    for scale_function in scale_functions:
        distance = AdaptiveAggregatedDistance(
            [distance0, distance1], scale_function=scale_function,
            initial_weights=initial_weights)
        distance.initialize(0, abc.sample_from_prior, x_0=x_0)
        val = distance(
            abc.sample_from_prior()[0], abc.sample_from_prior()[1], t=0)
        assert isinstance(val, float)
        assert (distance.weights[0] == initial_weights).all()
        distance.update(1, abc.sample_from_prior)
        assert (distance.weights[1] != distance.weights[0]).all()


def test_normalkernel():
    x0 = {'y0': np.array([1, 0]), 'y1': 1}
    x = {'y0': np.array([2, 2]), 'y1': 2}

    # use default cov
    kernel = NormalKernel()
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    # expected value
    logterm = 3 * np.log(2 * np.pi * 1)
    quadterm = 1**2 + 2**2 + 1**2
    expected = - 0.5 * (logterm + quadterm)
    assert np.isclose(ret, expected)

    # define own cov
    cov = np.array([[2, 1, 0], [0, 2, 1], [0, 1, 3]])
    kernel = NormalKernel(cov=cov)
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = sp.stats.multivariate_normal(cov=cov).logpdf([1, 2, 1])
    assert np.isclose(ret, expected)

    # define own keys, linear output
    kernel = NormalKernel(keys=['y0'], ret_scale=SCALE_LIN)
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = sp.stats.multivariate_normal(cov=np.eye(2)).pdf([1, 2])
    assert np.isclose(ret, expected)


def test_independentnormalkernel():
    x0 = {'y0': np.array([1, 2]), 'y1': 2.5}
    x = {'y0': np.array([0, 0]), 'y1': 7}

    # use default var
    kernel = IndependentNormalKernel()
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = -0.5 * (3 * np.log(2 * np.pi * 1) + 1**2 + 2**2 + 4.5**2)
    sp_expected = np.log(
        np.prod([sp.stats.norm.pdf(x=x, loc=0, scale=1)
                 for x in [1, 2, 4.5]]))
    assert np.isclose(expected, sp_expected)
    assert np.isclose(ret, expected)

    # define own var
    kernel = IndependentNormalKernel([1, 2, 3])
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = -0.5 * (3 * np.log(2 * np.pi) + np.log(1) + np.log(2)
                       + np.log(3) + 1**2 / 1 + 2**2 / 2 + 4.5**2 / 3)
    sp_expected = np.log(
        np.prod([sp.stats.norm.pdf(x=x, loc=0, scale=s)
                 for x, s in zip([1, 2, 4.5], np.sqrt([1, 2, 3]))]))
    assert np.isclose(expected, sp_expected)
    assert np.isclose(ret, expected)

    # compare to normal kernel
    normal_kernel = NormalKernel(cov=np.diag([1, 2, 3]))
    normal_kernel.initialize(0, None, x0)
    normal_ret = normal_kernel(x, x0)
    assert np.isclose(ret, normal_ret)

    # function var
    def var(p):
        if p is None:
            return 42
        return np.array([p['th0'], p['th1'], 3])

    kernel = IndependentNormalKernel(var)
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0, par={'th0': 1, 'th1': 2})
    assert np.isclose(ret, expected)


def test_independentlaplacekernel():
    x0 = {'y0': np.array([1, 2]), 'y1': 2.5}
    x = {'y0': np.array([0, 0]), 'y1': 7}

    # use default var
    kernel = IndependentLaplaceKernel()
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = - (3 * np.log(2 * 1) + 1 + 2 + 4.5)
    sp_expected = np.log(
        np.prod([sp.stats.laplace.pdf(x=x, loc=0, scale=1)
                 for x in [1, 2, 4.5]]))
    assert np.isclose(expected, sp_expected)
    assert np.isclose(ret, expected)

    # define own var
    kernel = IndependentLaplaceKernel([1, 2, 3])
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = - (np.log(2 * 1) + np.log(2 * 2) + np.log(2 * 3)
                  + 1 / 1 + 2 / 2 + 4.5 / 3)
    sp_expected = np.log(
        np.prod([sp.stats.laplace.pdf(x=x, loc=0, scale=s)
                 for x, s in zip([1, 2, 4.5], [1, 2, 3])]))
    assert np.isclose(expected, sp_expected)
    assert np.isclose(ret, expected)

    # function var
    def var(par):
        return np.array([par['th0'], par['th1'], 3])

    kernel = IndependentLaplaceKernel(var)
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0, par={'th0': 1, 'th1': 2})
    assert np.isclose(ret, expected)


def test_binomialkernel():
    x0 = {'y0': np.array([4, 5]), 'y1': 7}
    x = {'y0': np.array([7, 7]), 'y1': 7}

    kernel = BinomialKernel(p=0.9)
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = np.sum(sp.stats.binom.logpmf(k=[4, 5, 7], n=[7, 7, 7], p=0.9))
    assert np.isclose(ret, expected)

    # 0 likelihood
    ret = kernel(x, {'y0': np.array([4, 10]), 'y1': 7})
    assert np.isclose(ret, -np.inf)

    # linear output
    kernel = BinomialKernel(p=0.9, ret_scale=SCALE_LIN)
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = np.prod(sp.stats.binom.pmf(k=[4, 5, 7], n=[7, 7, 7], p=0.9))
    assert np.isclose(ret, expected)

    # function p
    def p(par):
        return np.array([0.9, 0.8, 0.7])
    kernel = BinomialKernel(p=p)
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = np.sum(sp.stats.binom.logpmf(
        k=[4, 5, 7], n=[7, 7, 7], p=[0.9, 0.8, 0.7]))
    assert np.isclose(ret, expected)


def test_poissonkernel():
    x0 = {'y0': np.array([4, 5]), 'y1': 7}
    x = {'y0': np.array([7, 7]), 'y1': 7}

    kernel = PoissonKernel()
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = np.sum(sp.stats.poisson.logpmf(k=[4, 5, 7], mu=[7, 7, 7]))
    assert np.isclose(ret, expected)

    # linear output
    kernel = PoissonKernel(ret_scale=SCALE_LIN)
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = np.prod(sp.stats.poisson.pmf(k=[4, 5, 7], mu=[7, 7, 7]))
    assert np.isclose(ret, expected)


def test_negativebinomialkernel():
    x0 = {'y0': np.array([4, 5]), 'y1': 8}
    x = {'y0': np.array([7, 7]), 'y1': 7}

    kernel = NegativeBinomialKernel(p=0.9)
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = np.sum(sp.stats.nbinom.logpmf(k=[4, 5, 8], n=[7, 7, 7], p=0.9))
    assert np.isclose(ret, expected)

    # linear output
    kernel = NegativeBinomialKernel(p=0.9, ret_scale=SCALE_LIN)
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = np.prod(sp.stats.nbinom.pmf(k=[4, 5, 8], n=[7, 7, 7], p=0.9))
    assert np.isclose(ret, expected)

    # function p
    def p(par):
        return np.array([0.9, 0.8, 0.7])
    kernel = NegativeBinomialKernel(p=p)
    kernel.initialize(0, None, x0)
    ret = kernel(x, x0)
    expected = np.sum(sp.stats.nbinom.logpmf(
        k=[4, 5, 8], n=[7, 7, 7], p=[0.9, 0.8, 0.7]))
    assert np.isclose(ret, expected)


def test_store_weights():
    """Test whether storing distance weights works."""
    abc = MockABC([{'s1': -1, 's2': -1, 's3': -1},
                   {'s1': -1, 's2': 0, 's3': 1}])
    x_0 = {'s1': 0, 's2': 0, 's3': 1}

    weights_file = tempfile.mkstemp(suffix=".json")[1]
    print(weights_file)

    def distance0(x, x_0):
        return abs(x['s1'] - x_0['s1'])

    def distance1(x, x_0):
        return np.sqrt((x['s2'] - x_0['s2'])**2)

    for distance in [AdaptivePNormDistance(log_file=weights_file),
                     AdaptiveAggregatedDistance(
                         [distance0, distance1], log_file=weights_file)]:
        distance.initialize(0, abc.sample_from_prior, x_0=x_0)
        distance.update(1, abc.sample_from_prior)
        distance.update(2, abc.sample_from_prior)

        weights = load_dict_from_json(weights_file)
        assert set(weights.keys()) == {0, 1, 2}

        expected = distance.weights
        for key, val in expected.items():
            if isinstance(val, np.ndarray):
                expected[key] = val.tolist()
        assert weights == expected
