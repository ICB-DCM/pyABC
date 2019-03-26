import multiprocessing
import pytest
import scipy as sp
import scipy.stats as st
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pyabc import (ABCSMC, RV, Distribution,
                   MedianEpsilon,
                   PercentileDistanceFunction, SimpleModel,
                   ConstantPopulationSize)
from pyabc.sampler import (Sample,
                           SingleCoreSampler,
                           MappingSampler,
                           MulticoreParticleParallelSampler,
                           DaskDistributedSampler,
                           ConcurrentFutureSampler,
                           MulticoreEvalParallelSampler,
                           RedisEvalParallelSamplerServerStarter)
from pyabc.population import Particle


def multi_proc_map(f, x):
    with multiprocessing.Pool() as pool:
        res = pool.map(f, x)
    return res


class GenericFutureWithProcessPool(ConcurrentFutureSampler):
    def __init__(self, map_=None):
        cfuture_executor = ProcessPoolExecutor(max_workers=8)
        client_max_jobs = 8
        super().__init__(cfuture_executor, client_max_jobs)


class GenericFutureWithProcessPoolBatch(ConcurrentFutureSampler):
    def __init__(self, map_=None):
        cfuture_executor = ProcessPoolExecutor(max_workers=8)
        client_max_jobs = 8
        batchsize = 15
        super().__init__(cfuture_executor, client_max_jobs,
                         batchsize=batchsize)


class GenericFutureWithThreadPool(ConcurrentFutureSampler):
    def __init__(self, map_=None):
        cfuture_executor = ThreadPoolExecutor(max_workers=8)
        client_max_jobs = 8
        super().__init__(cfuture_executor, client_max_jobs)


class MultiProcessingMappingSampler(MappingSampler):
    def __init__(self, map_=None):
        super().__init__(multi_proc_map)


class DaskDistributedSamplerBatch(DaskDistributedSampler):
    def __init__(self, map_=None):
        batchsize = 20
        super().__init__(batchsize=batchsize)


class WrongOutputSampler(SingleCoreSampler):
    def sample_until_n_accepted(self, n, simulate_one):
        return super().sample_until_n_accepted(n + 1, simulate_one)


def RedisEvalParallelSamplerServerStarterWrapper():
    return RedisEvalParallelSamplerServerStarter(batch_size=5)


@pytest.fixture(params=[SingleCoreSampler,
                        RedisEvalParallelSamplerServerStarterWrapper,
                        MulticoreEvalParallelSampler,
                        MultiProcessingMappingSampler,
                        MulticoreParticleParallelSampler,
                        MappingSampler,
                        DaskDistributedSampler,
                        DaskDistributedSamplerBatch,
                        GenericFutureWithThreadPool,
                        GenericFutureWithProcessPool,
                        GenericFutureWithProcessPoolBatch
                        ])
def sampler(request):
    s = request.param()
    yield s
    try:
        s.cleanup()
    except AttributeError:
        pass


@pytest.fixture
def redis_starter_sampler():
    s = RedisEvalParallelSamplerServerStarter(batch_size=5)
    yield s
    s.cleanup()


def test_two_competing_gaussians_multiple_population(db_path, sampler):
    two_competing_gaussians_multiple_population(
        db_path, sampler, 1)


def test_two_competing_gaussians_multiple_population_2_evaluations(
        db_path, redis_starter_sampler):
    two_competing_gaussians_multiple_population(db_path,
                                                redis_starter_sampler, 2)


def two_competing_gaussians_multiple_population(db_path, sampler, n_sim):
    # Define a gaussian model
    sigma = .5

    def model(args):
        return {"y": st.norm(args['x'], sigma).rvs()}

    # We define two models, but they are identical so far
    models = [model, model]
    models = list(map(SimpleModel, models))

    # However, our models' priors are not the same. Their mean differs.
    mu_x_1, mu_x_2 = 0, 1
    parameter_given_model_prior_distribution = [
        Distribution(x=RV("norm", mu_x_1, sigma)),
        Distribution(x=RV("norm", mu_x_2, sigma))
    ]

    # We plug all the ABC setup together
    nr_populations = 2
    pop_size = ConstantPopulationSize(40, nr_samples_per_parameter=n_sim)
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 PercentileDistanceFunction(measures_to_use=["y"]),
                 pop_size,
                 eps=MedianEpsilon(),
                 sampler=sampler)

    # Finally we add meta data such as model names and
    # define where to store the results
    # y_observed is the important piece here: our actual observation.
    y_observed = 1
    abc.new(db_path, {"y": y_observed})

    # We run the ABC with 3 populations max
    minimum_epsilon = .05
    history = abc.run(minimum_epsilon, max_nr_populations=nr_populations)

    # Evaluate the model probabililties
    mp = history.get_model_probabilities(history.max_t)

    def p_y_given_model(mu_x_model):
        res = st.norm(mu_x_model, sp.sqrt(sigma**2 + sigma**2)).pdf(y_observed)
        return res

    p1_expected_unnormalized = p_y_given_model(mu_x_1)
    p2_expected_unnormalized = p_y_given_model(mu_x_2)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized
                                              + p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized
                                              + p2_expected_unnormalized)
    assert history.max_t == nr_populations-1
    # the next line only tests if we obtain correct numerical types
    try:
        mp0 = mp.p[0]
    except KeyError:
        mp0 = 0

    try:
        mp1 = mp.p[1]
    except KeyError:
        mp1 = 0

    assert abs(mp0 - p1_expected) + abs(mp1 - p2_expected) < sp.inf


def test_in_memory(redis_starter_sampler):
    db_path = "sqlite://"
    two_competing_gaussians_multiple_population(db_path,
                                                redis_starter_sampler, 1)


def test_wrong_output_sampler():
    sampler = WrongOutputSampler()
    def simulate_one():
        return Particle(m=0, parameter={}, weight=0,
                        accepted_sum_stats=[], accepted_distances=[],
                        accepted=True)
    with pytest.raises(AssertionError):
        sampler.sample_until_n_accepted(5, simulate_one)
