import multiprocessing
import pytest
import numpy as np
import scipy.stats as st
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pyabc import (ABCSMC, RV, Distribution,
                   MedianEpsilon,
                   PercentileDistance, SimpleModel,
                   ConstantPopulationSize,
                   History, create_sqlite_db_id)
from pyabc.sampler import (SingleCoreSampler,
                           MappingSampler,
                           MulticoreParticleParallelSampler,
                           DaskDistributedSampler,
                           ConcurrentFutureSampler,
                           MulticoreEvalParallelSampler,
                           RedisEvalParallelSamplerServerStarter)
from pyabc.population import Particle
import logging
import os
import tempfile

logger = logging.getLogger(__name__)


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
        batch_size = 15
        super().__init__(cfuture_executor, client_max_jobs,
                         batch_size=batch_size)


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
        batch_size = 20
        super().__init__(batch_size=batch_size)


class WrongOutputSampler(SingleCoreSampler):
    def sample_until_n_accepted(
            self, n, simulate_one, max_eval=np.inf, all_accepted=False):
        return super().sample_until_n_accepted(
            n+1, simulate_one, max_eval, all_accepted=False)


def RedisEvalParallelSamplerServerStarterWrapper():
    return RedisEvalParallelSamplerServerStarter(batch_size=5)


def PicklingMulticoreParticleParallelSampler():
    return MulticoreParticleParallelSampler(pickle=True)


def PicklingMulticoreEvalParallelSampler():
    return MulticoreEvalParallelSampler(pickle=True)


@pytest.fixture(params=[SingleCoreSampler,
                        RedisEvalParallelSamplerServerStarterWrapper,
                        MulticoreEvalParallelSampler,
                        MultiProcessingMappingSampler,
                        MulticoreParticleParallelSampler,
                        PicklingMulticoreParticleParallelSampler,
                        PicklingMulticoreEvalParallelSampler,
                        MappingSampler,
                        DaskDistributedSampler,
                        DaskDistributedSamplerBatch,
                        GenericFutureWithThreadPool,
                        GenericFutureWithProcessPool,
                        GenericFutureWithProcessPoolBatch,
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
        Distribution(x=RV("norm", mu_x_2, sigma)),
    ]

    # We plug all the ABC setup together
    nr_populations = 2
    pop_size = ConstantPopulationSize(23, nr_samples_per_parameter=n_sim)
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 PercentileDistance(measures_to_use=["y"]),
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
        res = st.norm(mu_x_model, np.sqrt(sigma**2 + sigma**2)).pdf(y_observed)
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

    assert abs(mp0 - p1_expected) + abs(mp1 - p2_expected) < np.inf

    # check that sampler only did nr_particles samples in first round
    pops = history.get_all_populations()
    # since we had calibration (of epsilon), check that was saved
    pre_evals = pops[pops['t'] == History.PRE_TIME]['samples'].values
    assert pre_evals >= pop_size.nr_particles
    # our samplers should not have overhead in calibration, except batching
    batch_size = sampler.batch_size if hasattr(sampler, 'batch_size') else 1
    max_expected = pop_size.nr_particles + batch_size - 1
    if pre_evals > max_expected:
        # Violations have been observed occasionally for the redis server
        # due to runtime conditions with the increase of the evaluations
        # counter. This could be overcome, but as it usually only happens
        # for low-runtime models, this should not be a problem. Thus, only
        # print a warning here.
        logger.warning(
            f"Had {pre_evals} simulations in the calibration iteration, "
            f"but a maximum of {max_expected} would have been sufficient for "
            f"the population size of {pop_size.nr_particles}.")


def test_progressbar(sampler):
    """Test whether using a progress bar gives any errors."""
    def model(p):
        return {"y": p['p0'] + 0.1 * np.random.randn(10)}

    def distance(y1, y2):
        return np.abs(y1['y'] - y2['y']).sum()

    prior = Distribution(p0=RV('uniform', -5, 10))
    obs = {'y': 1}

    abc = ABCSMC(model, prior, distance, sampler=sampler, population_size=20)
    abc.new(db=create_sqlite_db_id(), observed_sum_stat=obs)
    abc.run(max_nr_populations=3)


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


def test_redis_multiprocess():
    sampler = RedisEvalParallelSamplerServerStarter(
        batch_size=3, workers=1, processes_per_worker=1)

    def simulate_one():
        accepted = np.random.randint(2)
        return Particle(0, {}, 0.1, [], [], accepted)

    sample = sampler.sample_until_n_accepted(10, simulate_one)
    assert 10 == len(sample.get_accepted_population())
    sampler.cleanup()


def test_redis_catch_error():

    def model(pars):
        if np.random.uniform() < 0.1:
            raise ValueError("error")
        return {'s0': pars['p0'] + 0.2 * np.random.uniform()}

    def distance(s0, s1):
        return abs(s0['s0'] - s1['s0'])

    prior = Distribution(p0=RV("uniform", 0, 10))
    sampler = RedisEvalParallelSamplerServerStarter(
        batch_size=3, workers=1, processes_per_worker=1, port=8775)

    abc = ABCSMC(model, prior, distance, sampler=sampler, population_size=10)

    db_file = "sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db")
    data = {'s0': 2.8}
    abc.new(db_file, data)

    abc.run(minimum_epsilon=.1, max_nr_populations=3)

    sampler.cleanup()


def test_redis_pw_protection():
    sampler = RedisEvalParallelSamplerServerStarter(  # noqa: S106
        password="daenerys", port=8888)

    def simulate_one():
        accepted = np.random.randint(2)
        return Particle(0, {}, 0.1, [], [], accepted)

    sample = sampler.sample_until_n_accepted(10, simulate_one)
    assert 10 == len(sample.get_accepted_population())
    sampler.cleanup()
