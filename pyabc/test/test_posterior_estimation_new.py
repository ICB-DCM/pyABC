import pytest
import os
import tempfile
import random
from pyabc import (ABCSMC, RV, ModelPerturbationKernel, Distribution,
                   MedianEpsilon, MinMaxDistanceFunction,
                   PercentileDistanceFunction, SimpleModel, Model, ModelResult,
                   MultivariateNormalTransition, ConstantPopulationStrategy)
from parallel.sampler import SingleCoreSampler
REMOVE_DB = True


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


def cookie_jar():
    def make_model(theta):
        def model(args):
            return {"result": 1 if random.random() > theta else 0}

        return model

    theta1 = .2
    theta2 = .6

    def cookie_jar_run(db_path, sampler):
        model1 = make_model(theta1)
        model2 = make_model(theta2)
        models = [model1, model2]
        models = list(map(SimpleModel, models))
        model_prior = RV("randint", 0, 2)
        population_size = ConstantPopulationStrategy(1500, 1)
        parameter_given_model_prior_distribution = [Distribution(), Distribution()]
        parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]
        abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.8),
                     parameter_given_model_prior_distribution,
                     parameter_perturbation_kernels,
                     MinMaxDistanceFunction(measures_to_use=["result"]),
                     MedianEpsilon(.1),
                     population_size,
                     sampler=sampler)

        options = {'db_path': db_path}
        abc.set_data({"result": 0}, 0, {}, options)

        minimum_epsilon = .2
        history = abc.run(minimum_epsilon)
        return history

    def cookie_jar_assert(history):
        mp = history.get_model_probabilities()
        expected_p1, expected_p2 = theta1 / (theta1 + theta2), theta2 / (theta1 + theta2)
        assert abs(mp.p[0] - expected_p1) + abs(mp.p[1] - expected_p2) < .05

    return cookie_jar_run, cookie_jar_assert


@pytest.fixture(params=[cookie_jar])
def abc_problem(request):
    return request.param()


@pytest.fixture(params=[SingleCoreSampler])
def sampler(request):
    return request.param()


def test_abc(db_path, sampler, abc_problem):
    runner, asserter = abc_problem
    result = runner(db_path, sampler)
    asserter(result)
