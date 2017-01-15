import multiprocessing
import os
import tempfile

import pytest
import scipy as sp
import scipy.stats as st

from pyabc import (ABCSMC, RV,  Distribution,
                   MedianEpsilon,
                   PercentileDistanceFunction, SimpleModel,
                    ConstantPopulationStrategy)
from pyabc.parallel import SingleCoreSampler, MappingSampler, MulticoreSampler

REMOVE_DB = False


def multi_proc_map(f, x):
    with multiprocessing.Pool() as pool:
        res = pool.map(f, x)
    return res


class MultiProcessingMappingSampler(MappingSampler):
    def __init__(self, map=None):
        super().__init__(multi_proc_map)


@pytest.fixture(params=[SingleCoreSampler, MultiProcessingMappingSampler, MulticoreSampler, MappingSampler])
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
    parameter_given_model_prior_distribution = [Distribution(x=RV("norm", mu_x_1, sigma)),
                                                Distribution(x=RV("norm", mu_x_2, sigma))]

    # We plug all the ABC setup together
    nr_populations = 3
    population_size = ConstantPopulationStrategy(40, 3)
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 PercentileDistanceFunction(measures_to_use=["y"]), population_size,
                 eps=MedianEpsilon(.2),
                 sampler=sampler)

    # Finally we add meta data such as model names and define where to store the results
    options = {'db_path': db_path}
    # y_observed is the important piece here: our actual observation.
    y_observed = 1
    abc.set_data({"y": y_observed}, 0, {}, options)

    # We run the ABC with 3 populations max
    minimum_epsilon = .05
    history = abc.run(minimum_epsilon)

    # Evaluate the model probabililties
    mp = history.get_model_probabilities(history.max_t)

    def p_y_given_model(mu_x_model):
        return st.norm(mu_x_model, sp.sqrt(sigma**2 + sigma**2)).pdf(y_observed)

    p1_expected_unnormalized = p_y_given_model(mu_x_1)
    p2_expected_unnormalized = p_y_given_model(mu_x_2)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    assert history.max_t == nr_populations-1
    # the next line only tests of we obtain correct numerical types
    assert abs(mp.p[0] - p1_expected) + abs(mp.p[1] - p2_expected) < sp.inf
