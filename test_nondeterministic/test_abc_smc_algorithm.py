"""
The tests here should try to test the correctness of the algorithm.
They are not intended to function primarily as integration tests (although
they certainly do partially). It is, e.g., not intended to test all possible
samplers with all possible problems here. The samplers have their own test.
"""

import os
import random
import tempfile

import pytest
import scipy as sp
import scipy.stats as st
from scipy.special import gamma, binom

from pyabc import (ABCSMC, RV, Distribution,
                   MedianEpsilon, MinMaxDistance,
                   PercentileDistance, SimpleModel, Model, ModelResult,
                   MultivariateNormalTransition, ConstantPopulationSize,
                   AdaptivePopulationSize, GridSearchCV)
from pyabc.sampler import MulticoreEvalParallelSampler
from pyabc.transition import LocalTransition

REMOVE_DB = False


@pytest.fixture(params=[LocalTransition, MultivariateNormalTransition])
def transition(request):
    return request.param


@pytest.fixture(params=[MulticoreEvalParallelSampler])
def sampler(request):
    s = request.param()
    yield s
    try:
        s.cleanup()
    except AttributeError:
        pass


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


def test_cookie_jar(db_path, sampler):
    def make_model(theta):
        def model(args):
            return {"result": 1 if random.random() > theta else 0}

        return model

    theta1 = .2
    theta2 = .6


    model1 = make_model(theta1)
    model2 = make_model(theta2)
    models = [model1, model2]
    models = list(map(SimpleModel, models))
    population_size = ConstantPopulationSize(1500)
    parameter_given_model_prior_distribution = [Distribution(), Distribution()]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["result"]),
                 population_size, eps=MedianEpsilon(.1), sampler=sampler)

    abc.new(db_path, {"result": 0})

    minimum_epsilon = .2
    history = abc.run(minimum_epsilon, max_nr_populations=1)

    mp = history.get_model_probabilities(history.max_t)
    expected_p1, expected_p2 = theta1 / (theta1 + theta2), theta2 / (theta1 +
                                                                     theta2)
    assert abs(mp.p[0] - expected_p1) + abs(mp.p[1] - expected_p2) < .05


def test_empty_population(db_path, sampler):
    def make_model(theta):
        def model(args):
            return {"result": 1 if random.random() > theta else 0}

        return model

    theta1 = .2
    theta2 = .6
    model1 = make_model(theta1)
    model2 = make_model(theta2)
    models = [model1, model2]
    models = list(map(SimpleModel, models))
    population_size = ConstantPopulationSize(1500)
    parameter_given_model_prior_distribution = [Distribution(), Distribution()]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["result"]),
                 population_size,
                 eps=MedianEpsilon(0),
                 sampler=sampler)
    abc.new(db_path, {"result": 0})

    minimum_epsilon = -1
    history = abc.run(minimum_epsilon, max_nr_populations=3)



    mp = history.get_model_probabilities(history.max_t)
    expected_p1, expected_p2 = theta1 / (theta1 + theta2), theta2 / (theta1 +
                                                                     theta2)
    assert abs(mp.p[0] - expected_p1) + abs(mp.p[1] - expected_p2) < .05


def test_beta_binomial_two_identical_models(db_path, sampler):
    binomial_n = 5

    def model_fun(args):
        return {"result": st.binom(binomial_n, args.theta).rvs()}

    models = [model_fun for _ in range(2)]
    models = list(map(SimpleModel, models))
    population_size = ConstantPopulationSize(800)
    parameter_given_model_prior_distribution = [Distribution(theta=st.beta(
                                                                      1, 1))
                                                for _ in range(2)]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["result"]),
                 population_size,
                 eps=MedianEpsilon(.1),
                 sampler=sampler)
    abc.new(db_path, {"result": 2})

    minimum_epsilon = .2
    history = abc.run(minimum_epsilon, max_nr_populations=3)
    mp = history.get_model_probabilities(history.max_t)
    assert abs(mp.p[0] - .5) + abs(mp.p[1] - .5) < .08


class AllInOneModel(Model):
    def summary_statistics(self, t, pars, sum_stats_calculator) -> ModelResult:
        return ModelResult(sum_stats={"result": 1})

    def accept(self, t, pars, sum_stats_calculator, distance_calculator,
               eps_calculator, acceptor, x_0) -> ModelResult:
        return ModelResult(accepted=True)


def test_all_in_one_model(db_path, sampler):
    models = [AllInOneModel() for _ in range(2)]
    population_size = ConstantPopulationSize(800)
    parameter_given_model_prior_distribution = [Distribution(theta=RV("beta",
                                                                      1, 1))
                                                for _ in range(2)]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["result"]),
                 population_size,
                 eps=MedianEpsilon(.1),
                 sampler=sampler)
    abc.new(db_path, {"result": 2})

    minimum_epsilon = .2
    history = abc.run(minimum_epsilon, max_nr_populations=3)
    mp = history.get_model_probabilities(history.max_t)
    assert abs(mp.p[0] - .5) + abs(mp.p[1] - .5) < .08


def test_beta_binomial_different_priors(db_path, sampler):
    binomial_n = 5

    def model(args):
        return {"result": st.binom(binomial_n, args['theta']).rvs()}

    models = [model for _ in range(2)]
    models = list(map(SimpleModel, models))
    population_size = ConstantPopulationSize(800)
    a1, b1 = 1, 1
    a2, b2 = 10, 1
    parameter_given_model_prior_distribution = [Distribution(theta=RV("beta",
                                                                      a1, b1)),
                                                Distribution(theta=RV("beta",
                                                                      a2, b2))]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["result"]),
                 population_size,
                 eps=MedianEpsilon(.1),
                 sampler=sampler)
    n1 = 2
    abc.new(db_path, {"result": n1})

    minimum_epsilon = .2
    history = abc.run(minimum_epsilon, max_nr_populations=3)
    mp = history.get_model_probabilities(history.max_t)

    def B(a, b):
        return gamma(a) * gamma(b) / gamma(a + b)

    def expected_p(a, b, n1):
        return binom(binomial_n, n1) * B(a + n1, b + binomial_n - n1) / B(a, b)

    p1_expected_unnormalized = expected_p(a1, b1, n1)
    p2_expected_unnormalized = expected_p(a2, b2, n1)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized +
                                              p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized +
                                              p2_expected_unnormalized)
    assert abs(mp.p[0] - p1_expected) + abs(mp.p[1] - p2_expected) < .08


def test_beta_binomial_different_priors_initial_epsilon_from_sample(db_path,
                                                                    sampler):
    binomial_n = 5

    def model(args):
        return {"result": st.binom(binomial_n, args.theta).rvs()}

    models = [model for _ in range(2)]
    models = list(map(SimpleModel, models))
    population_size = ConstantPopulationSize(800)
    a1, b1 = 1, 1
    a2, b2 = 10, 1
    parameter_given_model_prior_distribution = [Distribution(theta=RV("beta",
                                                                      a1, b1)),
                                                Distribution(theta=RV("beta",
                                                                      a2, b2))]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["result"]),
                 population_size,
                 eps=MedianEpsilon(median_multiplier=.9),
                 sampler=sampler)
    n1 = 2
    abc.new(db_path, {"result": n1})

    minimum_epsilon = -1
    history = abc.run(minimum_epsilon, max_nr_populations=5)
    mp = history.get_model_probabilities(history.max_t)

    def B(a, b):
        return gamma(a) * gamma(b) / gamma(a + b)

    def expected_p(a, b, n1):
        return binom(binomial_n, n1) * B(a + n1, b + binomial_n - n1) / B(a, b)

    p1_expected_unnormalized = expected_p(a1, b1, n1)
    p2_expected_unnormalized = expected_p(a2, b2, n1)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized +
                                              p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized +
                                              p2_expected_unnormalized)

    assert abs(mp.p[0] - p1_expected) + abs(mp.p[1] - p2_expected) < .08


def test_continuous_non_gaussian(db_path, sampler):
    def model(args):
        return {"result": np.random.rand() * args['u']}

    models = [model]
    models = list(map(SimpleModel, models))
    population_size = ConstantPopulationSize(250)
    parameter_given_model_prior_distribution = [Distribution(u=RV("uniform", 0,
                                                                  1))]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["result"]),
                 population_size,
                 eps=MedianEpsilon(.2),
                 sampler=sampler)
    d_observed = .5
    abc.new(db_path, {"result": d_observed})
    abc.do_not_stop_when_only_single_model_alive()

    minimum_epsilon = -1
    history = abc.run(minimum_epsilon, max_nr_populations=2)
    posterior_x, posterior_weight = history.get_distribution(0, None)
    posterior_x = posterior_x["u"].values
    sort_indices = np.argsort(posterior_x)
    f_empirical = np.interpolate.interp1d(np.hstack((-200,
                                                     posterior_x[sort_indices],
                                                     200)),
                                          np.hstack((0,
                                                     np.cumsum(
                                                         posterior_weight[
                                                             sort_indices]),
                                                     1)))

    @sp.vectorize
    def f_expected(u):
        return (np.log(u)-np.log(d_observed)) / (- np.log(d_observed)) * \
               (u > d_observed)

    x = np.linspace(0.1, 1)
    max_distribution_difference = np.absolute(f_empirical(x) -
                                              f_expected(x)).max()
    assert max_distribution_difference < 0.12


def mean_and_std(values, weights):
    mean = (values * weights).sum()
    std = np.sqrt(((values - mean)**2 * weights).sum())
    return mean, std


def test_gaussian_single_population(db_path, sampler):
    sigma_prior = 1
    sigma_ground_truth = 1
    observed_data = 1

    def model(args):
        return {"y": st.norm(args['x'], sigma_ground_truth).rvs()}

    models = [model]
    models = list(map(SimpleModel, models))
    nr_populations = 1
    population_size = ConstantPopulationSize(600)
    parameter_given_model_prior_distribution = [Distribution(x=RV("norm", 0,
                                                                  sigma_prior))
                                                ]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["y"]),
                 population_size,
                 eps=MedianEpsilon(.1),
                 sampler=sampler)
    abc.new(db_path, {"y": observed_data})

    minimum_epsilon = -1


    abc.do_not_stop_when_only_single_model_alive()
    history = abc.run(minimum_epsilon, max_nr_populations=nr_populations)
    posterior_x, posterior_weight = history.get_distribution(0, None)
    posterior_x = posterior_x["x"].values
    sort_indices = np.argsort(posterior_x)
    f_empirical = sp.interpolate.interp1d(np.hstack((-200, posterior_x[sort_indices], 200)),
                                          np.hstack((0, np.cumsum(posterior_weight[sort_indices]), 1)))

    sigma_x_given_y = 1 / np.sqrt(1 / sigma_prior**2 + 1 / sigma_ground_truth**2)
    mu_x_given_y = sigma_x_given_y**2 * observed_data / sigma_ground_truth**2
    expected_posterior_x = st.norm(mu_x_given_y, sigma_x_given_y)
    x = np.linspace(-8, 8)
    max_distribution_difference = np.absolute(f_empirical(x) - expected_posterior_x.cdf(x)).max()
    assert max_distribution_difference < 0.12
    assert history.max_t == nr_populations-1
    mean_emp, std_emp = mean_and_std(posterior_x, posterior_weight)
    assert abs(mean_emp - mu_x_given_y) < .07
    assert abs(std_emp - sigma_x_given_y) < .1


def test_gaussian_multiple_populations(db_path, sampler):
    sigma_x = 1
    sigma_y = .5
    y_observed = 2

    def model(args):
        return {"y": st.norm(args['x'], sigma_y).rvs()}

    models = [model]
    models = list(map(SimpleModel, models))
    nr_populations = 4
    population_size = ConstantPopulationSize(600)
    parameter_given_model_prior_distribution = [Distribution(x=st.norm(0, sigma_x))]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["y"]),
                 population_size,
                 eps=MedianEpsilon(.2),
                 sampler=sampler)

    abc.new(db_path, {"y": y_observed})

    minimum_epsilon = -1

    abc.do_not_stop_when_only_single_model_alive()
    history = abc.run(minimum_epsilon, max_nr_populations=nr_populations)
    posterior_x, posterior_weight = history.get_distribution(0, None)
    posterior_x = posterior_x["x"].values
    sort_indices = np.argsort(posterior_x)
    f_empirical = sp.interpolate.interp1d(np.hstack((-200, posterior_x[sort_indices], 200)),
                                          np.hstack((0, np.cumsum(posterior_weight[sort_indices]), 1)))

    sigma_x_given_y = 1 / np.sqrt(1 / sigma_x**2 + 1 / sigma_y**2)
    mu_x_given_y = sigma_x_given_y**2 * y_observed / sigma_y**2
    expected_posterior_x = st.norm(mu_x_given_y, sigma_x_given_y)
    x = np.linspace(-8, 8)
    max_distribution_difference = np.absolute(f_empirical(x) - expected_posterior_x.cdf(x)).max()
    assert max_distribution_difference < 0.052
    assert history.max_t == nr_populations-1
    mean_emp, std_emp = mean_and_std(posterior_x, posterior_weight)
    assert abs(mean_emp - mu_x_given_y) < .07
    assert abs(std_emp - sigma_x_given_y) < .12


def test_gaussian_multiple_populations_crossval_kde(db_path, sampler):
    sigma_x = 1
    sigma_y = .5
    y_observed = 2

    def model(args):
        return {"y": st.norm(args['x'], sigma_y).rvs()}

    models = [model]
    models = list(map(SimpleModel, models))
    nr_populations = 4
    population_size = ConstantPopulationSize(600)
    parameter_given_model_prior_distribution = [Distribution(x=st.norm(0, sigma_x))]
    parameter_perturbation_kernels = [GridSearchCV(MultivariateNormalTransition(),
                                      {"scaling": np.logspace(-1, 1.5, 5)})]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["y"]),
                 population_size,
                 transitions=parameter_perturbation_kernels,
                 eps=MedianEpsilon(.2),
                 sampler=sampler)
    abc.new(db_path, {"y": y_observed})

    minimum_epsilon = -1

    abc.do_not_stop_when_only_single_model_alive()
    history = abc.run(minimum_epsilon, max_nr_populations=nr_populations)
    posterior_x, posterior_weight = history.get_distribution(0, None)
    posterior_x = posterior_x["x"].values
    sort_indices = np.argsort(posterior_x)
    f_empirical = sp.interpolate.interp1d(np.hstack((-200, posterior_x[sort_indices], 200)),
                                          np.hstack((0, np.cumsum(posterior_weight[sort_indices]), 1)))

    sigma_x_given_y = 1 / np.sqrt(1 / sigma_x**2 + 1 / sigma_y**2)
    mu_x_given_y = sigma_x_given_y**2 * y_observed / sigma_y**2
    expected_posterior_x = st.norm(mu_x_given_y, sigma_x_given_y)
    x = np.linspace(-8, 8)
    max_distribution_difference = np.absolute(f_empirical(x) - expected_posterior_x.cdf(x)).max()
    assert max_distribution_difference < 0.052
    assert history.max_t == nr_populations-1
    mean_emp, std_emp = mean_and_std(posterior_x, posterior_weight)
    assert abs(mean_emp - mu_x_given_y) < .07
    assert abs(std_emp - sigma_x_given_y) < .12


def test_two_competing_gaussians_single_population(db_path, sampler, transition):
    sigma_x = .5
    sigma_y = .5
    y_observed = 1

    def model(args):
        return {"y": st.norm(args['x'], sigma_y).rvs()}

    models = [model, model]
    models = list(map(SimpleModel, models))
    population_size = ConstantPopulationSize(500)
    mu_x_1, mu_x_2 = 0, 1
    parameter_given_model_prior_distribution = [
        Distribution(x=st.norm(mu_x_1, sigma_x)),
        Distribution(x=st.norm(mu_x_2, sigma_x))]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["y"]),
                 population_size,
                 transitions=[transition(), transition()],
                 eps=MedianEpsilon(.02),
                 sampler=sampler)
    abc.new(db_path, {"y": y_observed})

    minimum_epsilon = -1
    nr_populations = 1
    abc.do_not_stop_when_only_single_model_alive()
    history = abc.run(minimum_epsilon, max_nr_populations=1)
    mp = history.get_model_probabilities(history.max_t)

    def p_y_given_model(mu_x_model):
        return st.norm(mu_x_model, np.sqrt(sigma_y**2+sigma_x**2)).pdf(y_observed)

    p1_expected_unnormalized = p_y_given_model(mu_x_1)
    p2_expected_unnormalized = p_y_given_model(mu_x_2)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    assert history.max_t == nr_populations - 1
    assert abs(mp.p[0] - p1_expected) + abs(mp.p[1] - p2_expected) < .07


def test_two_competing_gaussians_multiple_population(db_path, sampler, transition):
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
        Distribution(x=st.norm(mu_x_2, sigma))]

    # We plug all the ABC setup together
    nr_populations = 3
    population_size = ConstantPopulationSize(400)

    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 PercentileDistance(measures_to_use=["y"]), population_size,
                 eps=MedianEpsilon(.2),
                 transitions=[transition(), transition()],
                 sampler=sampler)

    # Finally we add meta data such as model names and define where to store the results
    # y_observed is the important piece here: our actual observation.
    y_observed = 1
    abc.new(db_path, {"y": y_observed})

    # We run the ABC with 3 populations max
    minimum_epsilon = .05
    history = abc.run(minimum_epsilon, max_nr_populations=nr_populations)

    # Evaluate the model probabililties
    mp = history.get_model_probabilities(history.max_t)

    def p_y_given_model(mu_x_model):
        return st.norm(mu_x_model, np.sqrt(sigma**2 + sigma**2)).pdf(y_observed)

    p1_expected_unnormalized = p_y_given_model(mu_x_1)
    p2_expected_unnormalized = p_y_given_model(mu_x_2)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    assert history.max_t == nr_populations-1
    assert abs(mp.p[0] - p1_expected) + abs(mp.p[1] - p2_expected) < .07


def test_empty_population_adaptive(db_path, sampler):
    def make_model(theta):
        def model(args):
            return {"result": 1 if random.random() > theta else 0}

        return model

    theta1 = .2
    theta2 = .6
    model1 = make_model(theta1)
    model2 = make_model(theta2)
    models = [model1, model2]
    models = list(map(SimpleModel, models))
    population_size = AdaptivePopulationSize(1500)
    parameter_given_model_prior_distribution = [Distribution(), Distribution()]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["result"]),
                 population_size,
                 eps=MedianEpsilon(0),
                 sampler=sampler)

    abc.new(db_path, {"result": 0})

    minimum_epsilon = -1
    history = abc.run(minimum_epsilon, max_nr_populations=3)
    mp = history.get_model_probabilities(history.max_t)
    expected_p1, expected_p2 = theta1 / (theta1 + theta2), theta2 / (theta1 + theta2)
    assert abs(mp.p[0] - expected_p1) + abs(mp.p[1] - expected_p2) < .1


def test_beta_binomial_two_identical_models_adaptive(db_path, sampler):
    binomial_n = 5

    def model_fun(args):
        return {"result": st.binom(binomial_n, args.theta).rvs()}

    models = [model_fun for _ in range(2)]
    models = list(map(SimpleModel, models))
    population_size = AdaptivePopulationSize(800)
    parameter_given_model_prior_distribution = [
        Distribution(theta=st.beta(1, 1)) for _ in range(2)]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["result"]),
                 population_size,
                 eps=MedianEpsilon(.1),
                 sampler=sampler)
    abc.new(db_path, {"result": 2})

    minimum_epsilon = .2
    history = abc.run(minimum_epsilon, max_nr_populations=3)
    mp = history.get_model_probabilities(history.max_t)
    assert abs(mp.p[0] - .5) + abs(mp.p[1] - .5) < .08


def test_gaussian_multiple_populations_adpative_population_size(db_path, sampler):
    sigma_x = 1
    sigma_y = .5
    y_observed = 2

    def model(args):
        return {"y": st.norm(args['x'], sigma_y).rvs()}

    models = [model]
    models = list(map(SimpleModel, models))
    nr_populations = 4
    population_size = AdaptivePopulationSize(600)
    parameter_given_model_prior_distribution = [
        Distribution(x=st.norm(0, sigma_x))]
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["y"]),
                 population_size,
                 eps=MedianEpsilon(.2),
                 sampler=sampler)
    abc.new(db_path, {"y": y_observed})

    minimum_epsilon = -1

    abc.do_not_stop_when_only_single_model_alive()
    history = abc.run(minimum_epsilon, max_nr_populations=nr_populations)
    posterior_x, posterior_weight = history.get_distribution(0, None)
    posterior_x = posterior_x["x"].values
    sort_indices = np.argsort(posterior_x)
    f_empirical = sp.interpolate.interp1d(np.hstack((-200, posterior_x[sort_indices], 200)),
                                          np.hstack((0, np.cumsum(posterior_weight[sort_indices]), 1)))

    sigma_x_given_y = 1 / np.sqrt(1 / sigma_x ** 2 + 1 / sigma_y ** 2)
    mu_x_given_y = sigma_x_given_y ** 2 * y_observed / sigma_y ** 2
    expected_posterior_x = st.norm(mu_x_given_y, sigma_x_given_y)
    x = np.linspace(-8, 8)
    max_distribution_difference = np.absolute(f_empirical(x) - expected_posterior_x.cdf(x)).max()
    assert max_distribution_difference < 0.15
    assert history.max_t == nr_populations - 1
    mean_emp, std_emp = mean_and_std(posterior_x, posterior_weight)
    assert abs(mean_emp - mu_x_given_y) < .07
    assert abs(std_emp - sigma_x_given_y) < .12


def test_two_competing_gaussians_multiple_population_adaptive_populatin_size(db_path, sampler):
    # Define a gaussian model
    sigma = .5

    def model(args):
        return {"y": st.norm(args['x'], sigma).rvs()}

    # We define two models, but they are identical so far
    models = [model, model]
    models = list(map(SimpleModel, models))

    # The prior over the model classes is uniform
    model_prior = RV("randint", 0, 2)

    # However, our models' priors are not the same. Their mean differs.
    mu_x_1, mu_x_2 = 0, 1
    parameter_given_model_prior_distribution = [Distribution(x=st.norm(mu_x_1, sigma)),
                                                Distribution(x=st.norm(mu_x_2, sigma))]

    # Particles are perturbed in a Gaussian fashion
    parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]

    # We plug all the ABC setup together
    nr_populations = 3
    population_size = AdaptivePopulationSize(400, mean_cv=0.05,
                                             max_population_size=1000)
    abc = ABCSMC(models, parameter_given_model_prior_distribution,
                 MinMaxDistance(measures_to_use=["y"]),
                 population_size,
                 model_prior=model_prior,
                 eps=MedianEpsilon(.2),
                 sampler=sampler)

    # Finally we add meta data such as model names and define where to store the results
    # y_observed is the important piece here: our actual observation.
    y_observed = 1
    abc.new(db_path, {"y": y_observed})

    # We run the ABC with 3 populations max
    minimum_epsilon = .05
    history = abc.run(minimum_epsilon, max_nr_populations=3)

    # Evaluate the model probabililties
    mp = history.get_model_probabilities(history.max_t)

    def p_y_given_model(mu_x_model):
        return st.norm(mu_x_model, np.sqrt(sigma ** 2 + sigma ** 2)).pdf(y_observed)

    p1_expected_unnormalized = p_y_given_model(mu_x_1)
    p2_expected_unnormalized = p_y_given_model(mu_x_2)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    assert history.max_t == nr_populations-1
    assert abs(mp.p[0] - p1_expected) + abs(mp.p[1] - p2_expected) < .07
