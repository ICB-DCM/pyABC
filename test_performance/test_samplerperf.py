import multiprocessing
import os
import tempfile
import time
import pytest
import numpy as np
import scipy.stats as st

from pyabc import (ABCSMC, RV, Distribution,
                   MedianEpsilon,
                   PercentileDistance, SimpleModel,
                   ConstantPopulationSize)
from pyabc.sampler import SingleCoreSampler, MappingSampler, MulticoreEvalParallelSampler, DaskDistributedSampler, \
                           ConcurrentFutureSampler
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

REMOVE_DB = False


def multi_proc_map(f, x):
    with multiprocessing.Pool() as pool:
        res = pool.map(f, x)
    return res


class GenericFutureWithProcessPool(ConcurrentFutureSampler):
    def __init__(self, map=None):
        cfuture_executor = ProcessPoolExecutor(max_workers=8)
        client_core_load_factor = 1.0
        client_max_jobs = 8
        throttle_delay = 0.0
        super().__init__(cfuture_executor, client_core_load_factor,
                         client_max_jobs, throttle_delay)


class GenericFutureWithThreadPool(ConcurrentFutureSampler):
    def __init__(self, map=None):
        cfuture_executor = ThreadPoolExecutor(max_workers=8)
        client_core_load_factor = 1.0
        client_max_jobs = 8
        throttle_delay = 0.0
        super().__init__(cfuture_executor, client_core_load_factor,
                         client_max_jobs, throttle_delay)


class MultiProcessingMappingSampler(MappingSampler):
    def __init__(self, map=None):
        super().__init__(multi_proc_map)


class DaskDistributedSamplerBatch(DaskDistributedSampler):
    def __init__(self, map=None):
        batch_size = 10
        super().__init__(batch_size=batch_size)


class GenericFutureWithProcessPoolBatch(ConcurrentFutureSampler):
    def __init__(self, map=None):
        cfuture_executor = ProcessPoolExecutor(max_workers=8)
        client_max_jobs = 8
        batch_size = 10
        super().__init__(cfuture_executor=cfuture_executor,
                         client_max_jobs=client_max_jobs,
                         batch_size=batch_size)


@pytest.fixture(params=[GenericFutureWithProcessPoolBatch,
                        DaskDistributedSamplerBatch,
                        MulticoreEvalParallelSampler, SingleCoreSampler,
                        MultiProcessingMappingSampler])
def sampler(request):
    return request.param()


@pytest.fixture
def db_path():
    db_file_location = os.path.join(tempfile.gettempdir(), "abc_unittest.db")
    db = "sqlite:///" + db_file_location
    yield db
    if REMOVE_DB:
        try:
            if REMOVE_DB:
                os.remove(db_file_location)
        except FileNotFoundError:
            pass


def test_two_competing_gaussians_multiple_population(db_path, sampler):
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
        Distribution(x=st.norm(mu_x_1, sigma)),
        Distribution(x=st.norm(mu_x_2, sigma))
    ]

    # We plug all the ABC setup together
    nr_populations = 1
    population_size = ConstantPopulationSize(20)
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 PercentileDistance(measures_to_use=["y"]),
                 population_size,
                 eps=MedianEpsilon(.2),
                 sampler=sampler)

    # Finally we add meta data such as model names and
    # define where to store the results
    # y_observed is the important piece here: our actual observation.
    y_observed = 2
    abc.new(db_path, {"y": y_observed})

    # We run the ABC with 3 populations max
    minimum_epsilon = .05
    history = abc.run(minimum_epsilon, max_nr_populations=nr_populations)

    # Evaluate the model probabililties
    history.get_model_probabilities(history.max_t)

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
