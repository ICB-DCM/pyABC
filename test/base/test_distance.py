import os
import tempfile

import numpy as np
import pytest
import scipy.linalg as la
import scipy.spatial.distance as sp_dist
import scipy.stats as stats

from pyabc.distance import (
    SCALE_LIN,
    AdaptiveAggregatedDistance,
    AdaptivePNormDistance,
    AggregatedDistance,
    BinomialKernel,
    IndependentLaplaceKernel,
    IndependentNormalKernel,
    InfoWeightedPNormDistance,
    MinMaxDistance,
    NegativeBinomialKernel,
    NormalKernel,
    PCADistance,
    PercentileDistance,
    PNormDistance,
    PoissonKernel,
    SlicedWassersteinDistance,
    WassersteinDistance,
    ZScoreDistance,
    bias,
    combined_mean_absolute_deviation,
    combined_median_absolute_deviation,
    mad_or_cmad,
    mean,
    mean_absolute_deviation,
    mean_absolute_deviation_to_observation,
    median,
    median_absolute_deviation,
    median_absolute_deviation_to_observation,
    root_mean_square_deviation,
    span,
    standard_deviation,
    standard_deviation_to_observation,
    std_or_rmsd,
)
from pyabc.inference import ABCSMC
from pyabc.parameters import Parameter
from pyabc.population import Particle, Sample
from pyabc.predictor import LinearPredictor
from pyabc.random_variables import RV, Distribution
from pyabc.storage import create_sqlite_db_id, load_dict_from_json
from pyabc.sumstat import Sumstat


class MockABC:
    def __init__(self, sumstats, accepted=None):
        self.sumstats = sumstats
        if accepted is None:
            accepted = [True] * len(sumstats)
        self.accepted_list = accepted

    def sample_from_prior(self) -> Sample:
        sample = Sample(record_rejected=True)
        for sumstat, accepted in zip(self.sumstats, self.accepted_list):
            sample.append(
                Particle(
                    m=0,
                    parameter=Parameter(
                        {'p1': np.random.randint(10), 'p2': np.random.randn()}
                    ),
                    weight=np.random.uniform(),
                    sum_stat=sumstat,
                    distance=np.random.uniform(),
                    accepted=accepted,
                ),
            )
        return sample


def test_single_parameter():
    dist_f = MinMaxDistance(measures_to_use=["a"])
    abc = MockABC([{"a": -3}, {"a": 3}, {"a": 10}])
    dist_f.initialize(0, abc.sample_from_prior, {}, 0)
    d = dist_f({"a": 1}, {"a": 2})
    assert 1.0 / 13 == d


def test_two_parameters_but_only_one_used():
    dist_f = MinMaxDistance(measures_to_use=["a"])
    abc = MockABC([{"a": -3, "b": 2}, {"a": 3, "b": 3}, {"a": 10, "b": 4}])
    dist_f.initialize(0, abc.sample_from_prior, {}, 0)
    d = dist_f({"a": 1, "b": 10}, {"a": 2, "b": 12})
    assert 1.0 / 13 == d


def test_two_parameters_and_two_used():
    dist_f = MinMaxDistance(measures_to_use=["a", "b"])
    abc = MockABC([{"a": -3, "b": 2}, {"a": 3, "b": 3}, {"a": 10, "b": 4}])
    dist_f.initialize(0, abc.sample_from_prior, {}, 0)
    d = dist_f({"a": 1, "b": 10}, {"a": 2, "b": 12})
    assert 1.0 / 13 + 2 / 2 == d


def test_single_parameter_percentile():
    dist_f = PercentileDistance(measures_to_use=["a"])
    abc = MockABC([{"a": -3}, {"a": 3}, {"a": 10}])
    dist_f.initialize(0, abc.sample_from_prior, {}, 0)
    d = dist_f({"a": 1}, {"a": 2})
    expected = 1 / (
        np.percentile([-3, 3, 10], 80) - np.percentile([-3, 3, 10], 20)
    )
    assert expected == d


def test_zscore_distance():
    """Test ZScoreDistance."""
    dist_f = ZScoreDistance()
    abc = MockABC([{"a": -3, "b": 2}, {"a": 3, "b": 3}, {"a": 10, "b": 4}])
    x0 = {"a": 7, "b": 3}
    n_y = len(x0)
    dist_f.initialize(0, abc.sample_from_prior, x0, 0)

    d = dist_f({"a": 4, "b": 2}, {"a": -5, "b": 10})
    expected = (abs((-5 - 4) / 5) + abs((10 - 2) / 10)) / n_y
    assert expected == d

    d = dist_f({"a": 4, "b": 2}, {"a": -5, "b": 0})
    assert np.inf == d

    d = dist_f({"a": 4, "b": 0}, {"a": -5, "b": 0})
    expected = (abs((-5 - 4) / 5)) / n_y
    assert expected == d


def test_pca_distance():
    """Test PCADistance."""
    dist_f = PCADistance()
    assert dist_f.requires_calibration()

    abc = MockABC(
        [{"a": -3.0, "b": 2.0}, {"a": 3.0, "b": 3.0}, {"a": 10.0, "b": 4.0}]
    )
    x0 = {"a": 7.0, "b": 3.0}
    dist_f.initialize(0, abc.sample_from_prior, x0, 0)
    assert dist_f.trafo.shape == (2, 2)

    # re-implement PCA whitening
    # data matrix, shape (n_sample, n_y)
    data = np.array([[-3.0, 2.0], [3.0, 3.0], [10.0, 4.0]])
    # covariance matrix
    cov = np.cov(data, rowvar=False, bias=False)
    # eigenvalues and eigenvectors
    eigval, eigvec = la.eigh(cov)
    # whitening transformation matrix
    mat = np.diag(1.0 / np.sqrt(eigval)) @ eigvec.T
    assert np.allclose(mat, dist_f.trafo)

    # check that converted data have standard distribution
    cov2 = np.cov(mat @ data.T, rowvar=True, bias=False)
    assert np.allclose(cov2, np.eye(2))


def test_pnormdistance():
    abc = MockABC(
        [{'s1': -1, 's2': -1, 's3': -1}, {'s1': -1, 's2': 0, 's3': 1}]
    )
    x_0 = {'s1': 0, 's2': 0, 's3': 1}

    # first test that for PNormDistance, the weights stay constant
    dist_f = PNormDistance(p=2)
    dist_f.initialize(0, abc.sample_from_prior, x_0=x_0, total_sims=0)

    # call distance function, also to initialize w
    d = dist_f(abc.sumstats[0], abc.sumstats[1], t=0)

    expected = pow(0**2 + 1**2 + 2**2, 1 / 2)
    assert expected == d

    assert dist_f.fixed_weights[0] == 1

    # maximum norm
    dist_f = PNormDistance(p=np.inf)
    dist_f.initialize(0, abc.sample_from_prior, x_0=x_0, total_sims=0)
    d = dist_f(abc.sumstats[0], abc.sumstats[1], t=0)
    assert d == 2


def test_adaptivepnormdistance():
    """
    Only tests basic running.
    """
    # TODO it could be checked that the scale functions lead to the expected
    #  values

    abc = MockABC(
        [{'s1': -1, 's2': -1, 's3': -1}, {'s1': -1, 's2': 0, 's3': 1}]
    )
    x_0 = {'s1': 0, 's2': 0, 's3': 1}

    scale_functions = [
        median_absolute_deviation,
        mean_absolute_deviation,
        standard_deviation,
        bias,
        root_mean_square_deviation,
        std_or_rmsd,
        median_absolute_deviation_to_observation,
        mean_absolute_deviation_to_observation,
        combined_median_absolute_deviation,
        mad_or_cmad,
        combined_mean_absolute_deviation,
        standard_deviation_to_observation,
    ]

    for scale_function in scale_functions:
        dist_f = AdaptivePNormDistance(scale_function=scale_function)
        dist_f.initialize(0, abc.sample_from_prior, x_0=x_0, total_sims=0)
        dist_f(abc.sumstats[0], abc.sumstats[1], t=0)
        assert (dist_f.scale_weights[0] != np.ones(3)).any()

    # test max weight ratio
    for scale_function in scale_functions:
        dist_f = AdaptivePNormDistance(
            scale_function=scale_function, max_scale_weight_ratio=20
        )
        dist_f.initialize(0, abc.sample_from_prior, x_0=x_0, total_sims=0)
        dist_f(abc.sumstats[0], abc.sumstats[1], t=0)

        weights = dist_f.scale_weights[0]
        assert (weights != np.ones(3)).any()
        assert np.max(weights) / np.min(weights[~np.isclose(weights, 0)]) <= 20


def test_adaptivepnorm_all_particles():
    """Test using rejected particles or not for weighting."""
    abc = MockABC(
        [
            {'s1': -1, 's2': -1, 's3': -1},
            {'s1': -1, 's2': 0, 's3': 1},
            {'s1': -2, 's2': 0.5, 's3': 3},
        ],
        accepted=[True, True, False],
    )
    x_0 = {'s1': 0, 's2': 0, 's3': 1}
    x_1 = {'s1': 0.5, 's2': 0.4, 's3': -5}

    # check that distance values calculated when using rejected particles
    #  or not differ

    dist_all = AdaptivePNormDistance(all_particles_for_scale=True)
    dist_all.initialize(0, abc.sample_from_prior, x_0=x_0, total_sims=0)

    dist_acc = AdaptivePNormDistance(all_particles_for_scale=False)
    dist_acc.initialize(0, abc.sample_from_prior, x_0=x_0, total_sims=0)

    assert dist_all(x_1, x_0, t=0) != dist_acc(x_1, x_0, t=0)


def test_scales():
    """Test scale functions directly."""
    scale_functions = [
        median_absolute_deviation,
        mean_absolute_deviation,
        standard_deviation,
        bias,
        root_mean_square_deviation,
        std_or_rmsd,
        median_absolute_deviation_to_observation,
        mean_absolute_deviation_to_observation,
        combined_median_absolute_deviation,
        mad_or_cmad,
        combined_mean_absolute_deviation,
        standard_deviation_to_observation,
    ]
    n_sample = 1000
    n_y = 50

    samples = np.random.normal(size=(n_sample, n_y))
    s0 = np.random.normal(size=(n_y,))
    s_ids = [f"s{ix}" for ix in range(n_y)]
    for scale in scale_functions:
        assert np.isfinite(scale(samples=samples, s0=s0, s_ids=s_ids)).all()

    samples[0, 0] = samples[1, 3] = samples[10, 2] = np.nan
    for scale in scale_functions:
        assert np.isfinite(scale(samples=samples, s0=s0, s_ids=s_ids)).all()

    s0_bad = np.random.normal(size=(n_y - 1,))
    for scale in scale_functions:
        with pytest.raises(AssertionError):
            scale(samples=samples, s0=s0_bad, s_ids=s_ids)

    s_ids_bad = [f"s{ix}" for ix in range(n_y + 1)]
    for scale in scale_functions:
        with pytest.raises(AssertionError):
            scale(samples=samples, s0=s0, s_ids=s_ids_bad)


def test_adaptivepnormdistance_initial_weights():
    abc = MockABC(
        [{'s1': -1, 's2': -1, 's3': -1}, {'s1': -1, 's2': 0, 's3': 1}]
    )
    x_0 = {'s1': 0, 's2': 0, 's3': 1}

    # first test that for PNormDistance, the weights stay constant
    initial_weights = {'s1': 1, 's2': 2, 's3': 3}
    dist_f = AdaptivePNormDistance(p=2, initial_scale_weights=initial_weights)
    dist_f.initialize(0, abc.sample_from_prior, x_0=x_0, total_sims=0)
    assert (dist_f.scale_weights[0] == np.array([1, 2, 3])).all()

    # call distance function
    d = dist_f(abc.sumstats[0], abc.sumstats[1], t=0)
    expected = pow(sum([(2 * 1) ** 2, (3 * 2) ** 2]), 1 / 2)
    assert expected == d

    # check updating works
    dist_f.update(1, abc.sample_from_prior, total_sims=0)
    assert (dist_f.scale_weights[1] != dist_f.scale_weights[0]).any()


def test_info_weighted_pnorm_distance():
    """Just test the info weighted distance pipeline."""
    db_file = create_sqlite_db_id()[len("sqlite:///") :]
    scale_log_file = tempfile.mkstemp()[1]
    info_log_file = tempfile.mkstemp()[1]
    info_sample_log_file = tempfile.mkstemp()[1]

    try:

        def model(p):
            return {
                "s0": p["p0"] + np.random.normal(),
                "s1": p["p1"] + np.random.normal(size=2),
            }

        prior = Distribution(p0=RV("uniform", 0, 1), p1=RV("uniform", 0, 10))
        data = {"s0": 0.5, "s1": np.array([5, 5])}

        for feature_normalization in ["mad", "std", "weights", "none"]:
            distance = InfoWeightedPNormDistance(
                predictor=LinearPredictor(),
                fit_info_ixs={1, 3},
                feature_normalization=feature_normalization,
                scale_log_file=scale_log_file,
                info_log_file=info_log_file,
                info_sample_log_file=info_sample_log_file,
            )
            abc = ABCSMC(model, prior, distance, population_size=100)
            abc.new("sqlite:///" + db_file, data)
            abc.run(max_nr_populations=3)
    finally:
        if os.path.exists(db_file):
            os.remove(db_file)
        if os.path.exists(scale_log_file):
            os.remove(scale_log_file)
        if os.path.exists(info_log_file):
            os.remove(info_log_file)
        # TODO remove info samples log files


def test_aggregateddistance():
    abc = MockABC([{'s0': -1, 's1': -1}, {'s0': -1, 's1': 0}])
    x_0 = {'s0': 0, 's1': 0}

    def distance0(x, x_0):
        return abs(x['s0'] - x_0['s0'])

    def distance1(x, x_0):
        return np.sqrt((x['s1'] - x_0['s1']) ** 2)

    distance = AggregatedDistance([distance0, distance1])
    distance.initialize(0, abc.sample_from_prior, x_0=x_0, total_sims=0)
    val = distance(abc.sumstats[0], abc.sumstats[1], t=0)
    assert isinstance(val, float)


def test_adaptiveaggregateddistance():
    abc = MockABC([{'s0': -1, 's1': -1}, {'s0': -1, 's1': 0}])
    x_0 = {'s0': 0, 's1': 0}

    def distance0(x, x_0):
        return abs(x['s0'] - x_0['s0'])

    def distance1(x, x_0):
        return np.sqrt((x['s1'] - x_0['s1']) ** 2)

    scale_functions = [span, mean, median]
    for scale_function in scale_functions:
        distance = AdaptiveAggregatedDistance(
            [distance0, distance1], scale_function=scale_function
        )
        distance.initialize(0, abc.sample_from_prior, x_0=x_0, total_sims=0)
        val = distance(abc.sumstats[0], abc.sumstats[1], t=0)
        assert isinstance(val, float)
        assert (distance.weights[0] != [1, 1]).any()


def test_adaptiveaggregateddistance_calibration():
    abc = MockABC([{'s0': -1, 's1': -1}, {'s0': -1, 's1': 0}])
    x_0 = {'s0': 0, 's1': 0}

    def distance0(x, x_0):
        return abs(x['s0'] - x_0['s0'])

    def distance1(x, x_0):
        return np.sqrt((x['s1'] - x_0['s1']) ** 2)

    scale_functions = [span, mean, median]
    initial_weights = np.array([2, 3])
    for scale_function in scale_functions:
        distance = AdaptiveAggregatedDistance(
            [distance0, distance1],
            scale_function=scale_function,
            initial_weights=initial_weights,
        )
        distance.initialize(0, abc.sample_from_prior, x_0=x_0, total_sims=0)
        val = distance(abc.sumstats[0], abc.sumstats[1], t=0)
        assert isinstance(val, float)
        assert (distance.weights[0] == initial_weights).all()
        distance.update(1, abc.sample_from_prior, total_sims=0)
        assert (distance.weights[1] != distance.weights[0]).all()


def test_normalkernel():
    x0 = {'y0': np.array([1, 0]), 'y1': 1}
    x = {'y0': np.array([2, 2]), 'y1': 2}

    # use default cov
    kernel = NormalKernel()
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    # expected value
    logterm = 3 * np.log(2 * np.pi * 1)
    quadterm = 1**2 + 2**2 + 1**2
    expected = -0.5 * (logterm + quadterm)
    assert np.isclose(ret, expected)

    # define own cov
    cov = np.array([[2, 1, 0], [0, 2, 1], [0, 1, 3]])
    kernel = NormalKernel(cov=cov)
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = stats.multivariate_normal(cov=cov).logpdf([1, 2, 1])
    assert np.isclose(ret, expected)

    # define own keys, linear output
    kernel = NormalKernel(keys=['y0'], ret_scale=SCALE_LIN)
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = stats.multivariate_normal(cov=np.eye(2)).pdf([1, 2])
    assert np.isclose(ret, expected)


def test_independentnormalkernel():
    x0 = {'y0': np.array([1, 2]), 'y1': 2.5}
    x = {'y0': np.array([0, 0]), 'y1': 7}

    # use default var
    kernel = IndependentNormalKernel()
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = -0.5 * (3 * np.log(2 * np.pi * 1) + 1**2 + 2**2 + 4.5**2)
    sp_expected = np.log(
        np.prod([stats.norm.pdf(x=x, loc=0, scale=1) for x in [1, 2, 4.5]])
    )
    assert np.isclose(expected, sp_expected)
    assert np.isclose(ret, expected)

    # define own var
    kernel = IndependentNormalKernel([1, 2, 3])
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = -0.5 * (
        3 * np.log(2 * np.pi)
        + np.log(1)
        + np.log(2)
        + np.log(3)
        + 1**2 / 1
        + 2**2 / 2
        + 4.5**2 / 3
    )
    sp_expected = np.log(
        np.prod(
            [
                stats.norm.pdf(x=x, loc=0, scale=s)
                for x, s in zip([1, 2, 4.5], np.sqrt([1, 2, 3]))
            ]
        )
    )
    assert np.isclose(expected, sp_expected)
    assert np.isclose(ret, expected)

    # compare to normal kernel
    normal_kernel = NormalKernel(cov=np.diag([1, 2, 3]))
    normal_kernel.initialize(0, None, x0, total_sims=0)
    normal_ret = normal_kernel(x, x0)
    assert np.isclose(ret, normal_ret)

    # function var
    def var(p):
        if p is None:
            return 42
        return np.array([p['th0'], p['th1'], 3])

    kernel = IndependentNormalKernel(var)
    kernel.initialize(0, None, x, total_sims=0)
    ret = kernel(x, x0, par={'th0': 1, 'th1': 2})
    assert np.isclose(ret, expected)


def test_independentlaplacekernel():
    x0 = {'y0': np.array([1, 2]), 'y1': 2.5}
    x = {'y0': np.array([0, 0]), 'y1': 7}

    # use default var
    kernel = IndependentLaplaceKernel()
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = -(3 * np.log(2 * 1) + 1 + 2 + 4.5)
    sp_expected = np.log(
        np.prod([stats.laplace.pdf(x=x, loc=0, scale=1) for x in [1, 2, 4.5]])
    )
    assert np.isclose(expected, sp_expected)
    assert np.isclose(ret, expected)

    # define own var
    kernel = IndependentLaplaceKernel([1, 2, 3])
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = -(
        np.log(2 * 1) + np.log(2 * 2) + np.log(2 * 3) + 1 / 1 + 2 / 2 + 4.5 / 3
    )
    sp_expected = np.log(
        np.prod(
            [
                stats.laplace.pdf(x=x, loc=0, scale=s)
                for x, s in zip([1, 2, 4.5], [1, 2, 3])
            ]
        )
    )
    assert np.isclose(expected, sp_expected)
    assert np.isclose(ret, expected)

    # function var
    def var(par):
        return np.array([par['th0'], par['th1'], 3])

    kernel = IndependentLaplaceKernel(var)
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0, par={'th0': 1, 'th1': 2})
    assert np.isclose(ret, expected)


def test_binomialkernel():
    x0 = {'y0': np.array([4, 5]), 'y1': 7}
    x = {'y0': np.array([7, 7]), 'y1': 7}

    kernel = BinomialKernel(p=0.9)
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = np.sum(stats.binom.logpmf(k=[4, 5, 7], n=[7, 7, 7], p=0.9))
    assert np.isclose(ret, expected)

    # 0 likelihood
    ret = kernel(x, {'y0': np.array([4, 10]), 'y1': 7})
    assert np.isclose(ret, -np.inf)

    # linear output
    kernel = BinomialKernel(p=0.9, ret_scale=SCALE_LIN)
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = np.prod(stats.binom.pmf(k=[4, 5, 7], n=[7, 7, 7], p=0.9))
    assert np.isclose(ret, expected)

    # function p
    def p(par):
        return np.array([0.9, 0.8, 0.7])

    kernel = BinomialKernel(p=p)
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = np.sum(
        stats.binom.logpmf(k=[4, 5, 7], n=[7, 7, 7], p=[0.9, 0.8, 0.7])
    )
    assert np.isclose(ret, expected)


def test_poissonkernel():
    x0 = {'y0': np.array([4, 5]), 'y1': 7}
    x = {'y0': np.array([7, 7]), 'y1': 7}

    kernel = PoissonKernel()
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = np.sum(stats.poisson.logpmf(k=[4, 5, 7], mu=[7, 7, 7]))
    assert np.isclose(ret, expected)

    # linear output
    kernel = PoissonKernel(ret_scale=SCALE_LIN)
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = np.prod(stats.poisson.pmf(k=[4, 5, 7], mu=[7, 7, 7]))
    assert np.isclose(ret, expected)


def test_negativebinomialkernel():
    x0 = {'y0': np.array([4, 5]), 'y1': 8}
    x = {'y0': np.array([7, 7]), 'y1': 7}

    kernel = NegativeBinomialKernel(p=0.9)
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = np.sum(stats.nbinom.logpmf(k=[4, 5, 8], n=[7, 7, 7], p=0.9))
    assert np.isclose(ret, expected)

    # linear output
    kernel = NegativeBinomialKernel(p=0.9, ret_scale=SCALE_LIN)
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = np.prod(stats.nbinom.pmf(k=[4, 5, 8], n=[7, 7, 7], p=0.9))
    assert np.isclose(ret, expected)

    # function p
    def p(par):
        return np.array([0.9, 0.8, 0.7])

    kernel = NegativeBinomialKernel(p=p)
    kernel.initialize(0, None, x0, total_sims=0)
    ret = kernel(x, x0)
    expected = np.sum(
        stats.nbinom.logpmf(k=[4, 5, 8], n=[7, 7, 7], p=[0.9, 0.8, 0.7])
    )
    assert np.isclose(ret, expected)


def test_store_weights():
    """Test whether storing distance weights works."""
    abc = MockABC(
        [{'s1': -1, 's2': -1, 's3': -1}, {'s1': -1, 's2': 0, 's3': 1}]
    )
    x_0 = {'s1': 0, 's2': 0, 's3': 1}

    weights_file = tempfile.mkstemp(suffix=".json")[1]
    print(weights_file)

    def distance0(x_, x_0_):
        return abs(x_['s1'] - x_0_['s1'])

    def distance1(x_, x_0_):
        return np.sqrt((x_['s2'] - x_0_['s2']) ** 2)

    for distance in [
        AdaptivePNormDistance(scale_log_file=weights_file),
        AdaptiveAggregatedDistance(
            [distance0, distance1, distance1], log_file=weights_file
        ),
    ]:
        distance.initialize(0, abc.sample_from_prior, x_0=x_0, total_sims=0)
        distance.update(1, abc.sample_from_prior, total_sims=0)
        distance.update(2, abc.sample_from_prior, total_sims=0)

        weights = load_dict_from_json(weights_file)
        assert set(weights.keys()) == {0, 1, 2}

        if isinstance(distance, AdaptivePNormDistance):
            expected = distance.scale_weights
        else:
            expected = distance.weights

        for key, val in expected.items():
            if isinstance(val, np.ndarray):
                expected[key] = val.tolist()
        for key, val in weights.items():
            if isinstance(val, dict):
                weights[key] = list(val.values())
        assert weights == expected

        os.remove(weights_file)


def test_wasserstein_distance():
    """Test Wasserstein and Sliced Wasserstein distances."""
    n_sample = 11

    def model_1d(p):
        return {"y": np.random.normal(p["p0"], 1.0, size=n_sample)}

    p_true = {"p0": -0.5}
    y0 = model_1d(p_true)

    p1 = {"p0": -0.55}
    y1 = model_1d(p1)

    p2 = {"p0": 3.55}
    y2 = model_1d(p2)

    class IdSumstat(Sumstat):
        """Identity summary statistic."""

        def __call__(self, data: dict) -> np.ndarray:
            # shape (n, dim)
            return data["y"].reshape((-1, 1))

    for p in [1, 2]:
        for distance in [
            WassersteinDistance(
                sumstat=IdSumstat(),
                p=p,
            ),
            SlicedWassersteinDistance(
                sumstat=IdSumstat(),
                p=p,
            ),
        ]:
            distance.initialize(x_0=y0)

            # evaluate distance
            dist = distance(y1, y0)

            assert dist > 0

            # sample from somewhere else
            assert dist < distance(y2, y0)

            # compare to ground truth
            if isinstance(distance, SlicedWassersteinDistance):
                continue

            # weights
            w = np.ones(shape=n_sample) / n_sample

            dist_exp = sp_dist.minkowski(
                np.sort(y1["y"].flatten()),
                np.sort(y0["y"]).flatten(),
                w=w,
                p=p,
            )

            assert np.isclose(dist, dist_exp)

    with pytest.raises(ValueError):
        WassersteinDistance(sumstat=IdSumstat(), p=3)

    # test integrated
    prior = Distribution(p0=RV("norm", 0, 2))
    db_file = tempfile.mkstemp(suffix=".db")[1]
    try:
        for distance in [
            WassersteinDistance(
                sumstat=IdSumstat(),
            ),
            SlicedWassersteinDistance(
                sumstat=IdSumstat(),
            ),
        ]:
            abc = ABCSMC(model_1d, prior, distance, population_size=10)
            abc.new("sqlite:///" + db_file, y0)
            abc.run(max_nr_populations=3)
    finally:
        os.remove(db_file)
