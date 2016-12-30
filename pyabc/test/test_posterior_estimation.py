import pytest
import os
import tempfile
import random
import scipy.stats as st
from pyabc import (ABCSMC, RV, ModelPerturbationKernel, Distribution,
                   MedianEpsilon, MinMaxDistanceFunction,
                   PercentileDistanceFunction, SimpleModel, Model, ModelResult,
                   MultivariateNormalTransition, ConstantPopulationStrategy,
                   AdaptivePopulationStrategy, GridSearchCV)
from parallel.sampler import SingleCoreSampler, MappingSampler
from scipy.special import gamma, binom
import scipy.interpolate
import scipy as sp
import multiprocessing
REMOVE_DB = False


class MultiProcessingMappingSampler(MappingSampler):
    def __init__(self, map=None):
        mapper = multiprocessing.Pool().map
        super().__init__(mapper)


@pytest.fixture(params=[SingleCoreSampler, MappingSampler, MultiProcessingMappingSampler])
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



    mp = history.get_model_probabilities(history.max_t)
    expected_p1, expected_p2 = theta1 / (theta1 + theta2), theta2 / (theta1 + theta2)
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
    model_prior = RV("randint", 0, 2)
    population_size = ConstantPopulationStrategy(1500, 3)
    parameter_given_model_prior_distribution = [Distribution(), Distribution()]
    parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.8),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 MinMaxDistanceFunction(measures_to_use=["result"]), MedianEpsilon(0),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    abc.set_data({"result": 0}, 0, {}, options)

    minimum_epsilon = -1
    history = abc.run(minimum_epsilon)



    mp = history.get_model_probabilities(history.max_t)
    expected_p1, expected_p2 = theta1 / (theta1 + theta2), theta2 / (theta1 + theta2)
    assert abs(mp.p[0] - expected_p1) + abs(mp.p[1] - expected_p2) < .05


def test_beta_binomial_two_identical_models(db_path, sampler):
    binomial_n = 5

    def model_fun(args):
        return {"result": st.binom(binomial_n, args.theta).rvs()}

    models = [model_fun for _ in range(2)]
    models = list(map(SimpleModel, models))
    model_prior = RV("randint", 0, 2)
    population_size = ConstantPopulationStrategy(800, 3)
    parameter_given_model_prior_distribution = [Distribution(theta=RV("beta", 1, 1))
                                                for _ in range(2)]
    parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.8),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 MinMaxDistanceFunction(measures_to_use=["result"]), MedianEpsilon(.1),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    abc.set_data({"result": 2}, 0, {}, options)

    minimum_epsilon = .2
    history = abc.run( minimum_epsilon)
    mp = history.get_model_probabilities(history.max_t)
    assert abs(mp.p[0] - .5) + abs(mp.p[1] - .5) < .08


class AllInOneModel(Model):
    def summary_statistics(self, pars, sum_stats_calculator) -> ModelResult:
        return ModelResult(sum_stats={"result": 1})

    def accept(self, pars, sum_stats_calculator, distance_calculator, eps) -> ModelResult:
        return ModelResult(accepted=True)


def test_all_in_one_model(db_path, sampler):
    models = [AllInOneModel() for _ in range(2)]
    model_prior = RV("randint", 0, 2)
    population_size = ConstantPopulationStrategy(800, 3)
    parameter_given_model_prior_distribution = [Distribution(theta=RV("beta", 1, 1))
                                                for _ in range(2)]
    parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.8),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 MinMaxDistanceFunction(measures_to_use=["result"]), MedianEpsilon(.1),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    abc.set_data({"result": 2}, 0, {}, options)

    minimum_epsilon = .2
    history = abc.run(minimum_epsilon)
    mp = history.get_model_probabilities(history.max_t)
    assert abs(mp.p[0] - .5) + abs(mp.p[1] - .5) < .08


def test_beta_binomial_different_priors(db_path, sampler):
    binomial_n = 5

    def model(args):
        return {"result": st.binom(binomial_n, args['theta']).rvs()}

    models = [model for _ in range(2)]
    models = list(map(SimpleModel, models))
    model_prior = RV("randint", 0, 2)
    population_size = ConstantPopulationStrategy(800, 3)
    a1, b1 = 1, 1
    a2, b2 = 10, 1
    parameter_given_model_prior_distribution = [Distribution(theta=RV("beta", a1, b1)),
                                                Distribution(theta=RV("beta", a2, b2))]
    parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.8),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 MinMaxDistanceFunction(measures_to_use=["result"]), MedianEpsilon(.1),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    n1 = 2
    abc.set_data({"result": n1}, 0, {}, options)

    minimum_epsilon = .2
    history = abc.run(minimum_epsilon)
    mp = history.get_model_probabilities(history.max_t)

    def B(a, b):
        return gamma(a) * gamma(b) / gamma(a + b)

    def expected_p(a, b, n1):
        return binom(binomial_n, n1) * B(a + n1, b + binomial_n - n1) / B(a, b)

    p1_expected_unnormalized = expected_p(a1, b1, n1)
    p2_expected_unnormalized = expected_p(a2, b2, n1)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized+p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized+p2_expected_unnormalized)
    assert abs(mp.p[0] - p1_expected) + abs(mp.p[1] - p2_expected) < .08


def test_beta_binomial_different_priors_initial_epsilon_from_sample(db_path, sampler):
    binomial_n = 5

    def model(args):
        return {"result": st.binom(binomial_n, args.theta).rvs()}

    models = [model for _ in range(2)]
    models = list(map(SimpleModel, models))
    model_prior = RV("randint", 0, 2)
    population_size = ConstantPopulationStrategy(800, 5)
    a1, b1 = 1, 1
    a2, b2 = 10, 1
    parameter_given_model_prior_distribution = [Distribution(theta=RV("beta", a1, b1)),
                                                Distribution(theta=RV("beta", a2, b2))]
    parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.8),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 MinMaxDistanceFunction(measures_to_use=["result"]),
                 MedianEpsilon(median_multiplier=.9), population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    n1 = 2
    abc.set_data({"result": n1}, 0, {}, options)

    minimum_epsilon = -1
    history = abc.run(minimum_epsilon)
    mp = history.get_model_probabilities(history.max_t)

    def B(a, b):
        return gamma(a) * gamma(b) / gamma(a + b)

    def expected_p(a, b, n1):
        return binom(binomial_n, n1) * B(a + n1, b + binomial_n - n1) / B(a, b)

    p1_expected_unnormalized = expected_p(a1, b1, n1)
    p2_expected_unnormalized = expected_p(a2, b2, n1)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)

    assert abs(mp.p[0] - p1_expected) + abs(mp.p[1] - p2_expected) < .08


def test_continuous_non_gaussian(db_path, sampler):
    def model(args):
        return {"result": sp.rand() * args['u']}

    models = [model]
    models = list(map(SimpleModel, models))
    model_prior = RV("randint", 0, 1)
    population_size = ConstantPopulationStrategy(250, 2)
    parameter_given_model_prior_distribution = [Distribution(u=RV("uniform", 0, 1))]
    parameter_perturbation_kernels = [MultivariateNormalTransition()]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(1, probability_to_stay=1),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 PercentileDistanceFunction(measures_to_use=["result"]), MedianEpsilon(.2),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    d_observed = .5
    abc.set_data({"result": d_observed}, 0, {}, options)
    abc.do_not_stop_when_only_single_model_alive()

    minimum_epsilon = -1
    history = abc.run(minimum_epsilon)
    posterior_x, posterior_weight = history.get_results_distribution(0, "u")
    sort_indices = sp.argsort(posterior_x)
    f_empirical = sp.interpolate.interp1d(sp.hstack((-200, posterior_x[sort_indices], 200)),
                                          sp.hstack((0, sp.cumsum(posterior_weight[sort_indices]), 1)))

    @sp.vectorize
    def f_expected(u):
        return (sp.log(u)-sp.log(d_observed)) / (- sp.log(d_observed)) * (u > d_observed)

    x = sp.linspace(0.1, 1)
    max_distribution_difference = sp.absolute(f_empirical(x) - f_expected(x)).max()
    assert max_distribution_difference < 0.12


def mean_and_std(values, weights):
    mean = (values * weights).sum()
    std = sp.sqrt(((values - mean)**2 * weights).sum())
    return mean, std


def test_gaussian_single_population(db_path, sampler):
    sigma_prior = 1
    sigma_ground_truth = 1
    observed_data = 1

    def model(args):
        return {"y": st.norm(args['x'], sigma_ground_truth).rvs()}

    models = [model]
    models = list(map(SimpleModel, models))
    model_prior = RV("randint", 0, 1)
    nr_populations = 1
    population_size = ConstantPopulationStrategy(600, nr_populations)
    parameter_given_model_prior_distribution = [Distribution(x=RV("norm", 0, sigma_prior))]
    parameter_perturbation_kernels = [MultivariateNormalTransition()]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(1, probability_to_stay=1),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 PercentileDistanceFunction(measures_to_use=["y"]), MedianEpsilon(.1),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    abc.set_data({"y": observed_data}, 0, {}, options)

    minimum_epsilon = -1


    abc.do_not_stop_when_only_single_model_alive()
    history = abc.run(minimum_epsilon)
    posterior_x, posterior_weight = history.get_results_distribution(0, "x")
    sort_indices = sp.argsort(posterior_x)
    f_empirical = sp.interpolate.interp1d(sp.hstack((-200, posterior_x[sort_indices], 200)),
                                          sp.hstack((0, sp.cumsum(posterior_weight[sort_indices]), 1)))

    sigma_x_given_y = 1 / sp.sqrt(1 / sigma_prior**2 + 1 / sigma_ground_truth**2)
    mu_x_given_y = sigma_x_given_y**2 * observed_data / sigma_ground_truth**2
    expected_posterior_x = st.norm(mu_x_given_y, sigma_x_given_y)
    x = sp.linspace(-8, 8)
    max_distribution_difference = sp.absolute(f_empirical(x) - expected_posterior_x.cdf(x)).max()
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
    model_prior = RV("randint", 0, 1)
    nr_populations = 4
    population_size = ConstantPopulationStrategy(600, nr_populations)
    parameter_given_model_prior_distribution = [Distribution(x=RV("norm", 0, sigma_x))]
    parameter_perturbation_kernels = [MultivariateNormalTransition()]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(1, probability_to_stay=1),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 PercentileDistanceFunction(measures_to_use=["y"]), MedianEpsilon(.2),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    abc.set_data({"y": y_observed}, 0, {}, options)

    minimum_epsilon = -1

    abc.do_not_stop_when_only_single_model_alive()
    history = abc.run(minimum_epsilon)
    posterior_x, posterior_weight = history.get_results_distribution(0, "x")
    sort_indices = sp.argsort(posterior_x)
    f_empirical = sp.interpolate.interp1d(sp.hstack((-200, posterior_x[sort_indices], 200)),
                                          sp.hstack((0, sp.cumsum(posterior_weight[sort_indices]), 1)))

    sigma_x_given_y = 1 / sp.sqrt(1 / sigma_x**2 + 1 / sigma_y**2)
    mu_x_given_y = sigma_x_given_y**2 * y_observed / sigma_y**2
    expected_posterior_x = st.norm(mu_x_given_y, sigma_x_given_y)
    x = sp.linspace(-8, 8)
    max_distribution_difference = sp.absolute(f_empirical(x) - expected_posterior_x.cdf(x)).max()
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
    model_prior = RV("randint", 0, 1)
    nr_populations = 4
    population_size = ConstantPopulationStrategy(600, nr_populations)
    parameter_given_model_prior_distribution = [Distribution(x=RV("norm", 0, sigma_x))]
    parameter_perturbation_kernels = [GridSearchCV(MultivariateNormalTransition(),
                                      {"scaling": sp.logspace(-1, 1.5, 5)})]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(1, probability_to_stay=1),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 PercentileDistanceFunction(measures_to_use=["y"]), MedianEpsilon(.2),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    abc.set_data({"y": y_observed}, 0, {}, options)

    minimum_epsilon = -1

    abc.do_not_stop_when_only_single_model_alive()
    history = abc.run(minimum_epsilon)
    posterior_x, posterior_weight = history.get_results_distribution(0, "x")
    sort_indices = sp.argsort(posterior_x)
    f_empirical = sp.interpolate.interp1d(sp.hstack((-200, posterior_x[sort_indices], 200)),
                                          sp.hstack((0, sp.cumsum(posterior_weight[sort_indices]), 1)))

    sigma_x_given_y = 1 / sp.sqrt(1 / sigma_x**2 + 1 / sigma_y**2)
    mu_x_given_y = sigma_x_given_y**2 * y_observed / sigma_y**2
    expected_posterior_x = st.norm(mu_x_given_y, sigma_x_given_y)
    x = sp.linspace(-8, 8)
    max_distribution_difference = sp.absolute(f_empirical(x) - expected_posterior_x.cdf(x)).max()
    assert max_distribution_difference < 0.052
    assert history.max_t == nr_populations-1
    mean_emp, std_emp = mean_and_std(posterior_x, posterior_weight)
    assert abs(mean_emp - mu_x_given_y) < .07
    assert abs(std_emp - sigma_x_given_y) < .12


def test_two_competing_gaussians_single_population(db_path, sampler):
    sigma_x = .5
    sigma_y = .5
    y_observed = 1

    def model(args):
        return {"y": st.norm(args['x'], sigma_y).rvs()}

    models = [model, model]
    models = list(map(SimpleModel, models))
    model_prior = RV("randint", 0, 2)
    population_size = ConstantPopulationStrategy(500, 1)
    mu_x_1, mu_x_2 = 0, 1
    parameter_given_model_prior_distribution = [Distribution(x=RV("norm", mu_x_1, sigma_x)),
                                                Distribution(x=RV("norm", mu_x_2, sigma_x))]
    parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.7),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 PercentileDistanceFunction(measures_to_use=["y"]), MedianEpsilon(.02),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    abc.set_data({"y": y_observed}, 0, {}, options)

    minimum_epsilon = -1
    nr_populations = 1
    abc.do_not_stop_when_only_single_model_alive()
    history = abc.run(minimum_epsilon)
    mp = history.get_model_probabilities(history.max_t)

    def p_y_given_model(mu_x_model):
        return st.norm(mu_x_model, sp.sqrt(sigma_y**2+sigma_x**2)).pdf(y_observed)

    p1_expected_unnormalized = p_y_given_model(mu_x_1)
    p2_expected_unnormalized = p_y_given_model(mu_x_2)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    assert history.max_t == nr_populations - 1
    assert abs(mp.p[0] - p1_expected) + abs(mp.p[1] - p2_expected) < .07


def test_two_competing_gaussians_multiple_population(db_path, sampler):
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
    parameter_given_model_prior_distribution = [Distribution(x=RV("norm", mu_x_1, sigma)),
                                                Distribution(x=RV("norm", mu_x_2, sigma))]

    # Particles are perturbed in a Gaussian fashion
    parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]

    # We plug all the ABC setup together
    nr_populations = 3
    population_size = ConstantPopulationStrategy(400, 3)
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.7),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 PercentileDistanceFunction(measures_to_use=["y"]), MedianEpsilon(.2),
                 population_size,
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
    model_prior = RV("randint", 0, 2)
    population_size = AdaptivePopulationStrategy(1500, 3)
    parameter_given_model_prior_distribution = [Distribution(), Distribution()]
    parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.8),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 MinMaxDistanceFunction(measures_to_use=["result"]), MedianEpsilon(0),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    abc.set_data({"result": 0}, 0, {}, options)

    minimum_epsilon = -1
    history = abc.run(minimum_epsilon)
    mp = history.get_model_probabilities(history.max_t)
    expected_p1, expected_p2 = theta1 / (theta1 + theta2), theta2 / (theta1 + theta2)
    assert abs(mp.p[0] - expected_p1) + abs(mp.p[1] - expected_p2) < .1


def test_beta_binomial_two_identical_models_adaptive(db_path, sampler):
    binomial_n = 5

    def model_fun(args):
        return {"result": st.binom(binomial_n, args.theta).rvs()}

    models = [model_fun for _ in range(2)]
    models = list(map(SimpleModel, models))
    model_prior = RV("randint", 0, 2)
    population_size = AdaptivePopulationStrategy(800, 3)
    parameter_given_model_prior_distribution = [Distribution(theta=RV("beta", 1, 1)) for _ in range(2)]
    parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.8),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 MinMaxDistanceFunction(measures_to_use=["result"]), MedianEpsilon(.1),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    abc.set_data({"result": 2}, 0, {}, options)

    minimum_epsilon = .2
    history = abc.run( minimum_epsilon)
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
    model_prior = RV("randint", 0, 1)
    nr_populations = 4
    population_size = AdaptivePopulationStrategy(600, nr_populations)
    parameter_given_model_prior_distribution = [Distribution(x=RV("norm", 0, sigma_x))]
    parameter_perturbation_kernels = [MultivariateNormalTransition()]
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(1, probability_to_stay=1),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 PercentileDistanceFunction(measures_to_use=["y"]), MedianEpsilon(.2),
                 population_size,
                 sampler=sampler)

    options = {'db_path': db_path}
    abc.set_data({"y": y_observed}, 0, {}, options)

    minimum_epsilon = -1

    abc.do_not_stop_when_only_single_model_alive()
    history = abc.run(minimum_epsilon)
    posterior_x, posterior_weight = history.get_results_distribution(0, "x")
    sort_indices = sp.argsort(posterior_x)
    f_empirical = sp.interpolate.interp1d(sp.hstack((-200, posterior_x[sort_indices], 200)),
                                          sp.hstack((0, sp.cumsum(posterior_weight[sort_indices]), 1)))

    sigma_x_given_y = 1 / sp.sqrt(1 / sigma_x ** 2 + 1 / sigma_y ** 2)
    mu_x_given_y = sigma_x_given_y ** 2 * y_observed / sigma_y ** 2
    expected_posterior_x = st.norm(mu_x_given_y, sigma_x_given_y)
    x = sp.linspace(-8, 8)
    max_distribution_difference = sp.absolute(f_empirical(x) - expected_posterior_x.cdf(x)).max()
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
    parameter_given_model_prior_distribution = [Distribution(x=RV("norm", mu_x_1, sigma)),
                                                Distribution(x=RV("norm", mu_x_2, sigma))]

    # Particles are perturbed in a Gaussian fashion
    parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]

    # We plug all the ABC setup together
    nr_populations = 3
    population_size = AdaptivePopulationStrategy(400, 3, mean_cv=0.05)
    abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.7),
                 parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                 PercentileDistanceFunction(measures_to_use=["y"]), MedianEpsilon(.2),
                 population_size,
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
        return st.norm(mu_x_model, sp.sqrt(sigma ** 2 + sigma ** 2)).pdf(y_observed)

    p1_expected_unnormalized = p_y_given_model(mu_x_1)
    p2_expected_unnormalized = p_y_given_model(mu_x_2)
    p1_expected = p1_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    p2_expected = p2_expected_unnormalized / (p1_expected_unnormalized + p2_expected_unnormalized)
    assert history.max_t == nr_populations-1
    assert abs(mp.p[0] - p1_expected) + abs(mp.p[1] - p2_expected) < .07
