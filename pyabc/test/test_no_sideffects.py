import unittest
from pyabc import (ABCSMC, RV, ModelPerturbationKernel, Distribution,
                    MedianEpsilon, Kernel,  PercentileDistanceFunction, SimpleModel)
import random
import scipy.stats as st
import scipy as sp
import os
import tempfile
import numpy as np


def set_seeds():
    random.seed(42)
    sp.random.seed(42)
    np.random.seed(42)


class TestNoSideEffects(unittest.TestCase):
    def setUp(self):
        self.db_file_location = os.path.join(tempfile.gettempdir(), "abc_unittest_db.db")
        self.db = "sqlite:///" + self.db_file_location
        self.clean_db()

    def clean_db(self):
        try:
            os.remove(self.db_file_location)
        except FileNotFoundError:
            pass

    def tearDown(self):
        self.clean_db()

    def test_no_side_effect_prior_sample(self):
        sigma_x = .5
        sigma_y = .5
        y_observed = 1

        def model(args):
            return {"y": st.norm(args['x'], sigma_y).rvs()}

        models = [model, model]
        models = list(map(SimpleModel, models))
        model_prior = RV("randint", 0, 2)
        nr_particles = 400
        mu_x_1, mu_x_2 = 0, 1
        parameter_given_model_prior_distribution = [Distribution(x=RV("norm", mu_x_1, sigma_x)),
                                                    Distribution(x=RV("norm", mu_x_2, sigma_x))]
        parameter_perturbation_kernels = [lambda t, stat: Kernel(stat['cov']) for _ in range(2)]
        abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.7),
                     parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                     PercentileDistanceFunction(measures_to_use=["y"]), MedianEpsilon(.2), nr_particles,
                     max_nr_allowed_sample_attempts_per_particle=2000)

        model_names = ["m1", "m2"]
        options = {'db_path': self.db}
        abc.set_data({"y": y_observed}, 0, {}, options, model_names)

        abc._points_sampled_from_prior = None
        set_seeds()
        result1 = abc.map_wrapper.wrap_map_sample_from_prior(abc)

        abc._points_sampled_from_prior = None
        set_seeds()
        result2 = abc.map_wrapper.wrap_map_sample_from_prior(abc)

        self.assertEqual(result1, result2)

    def test_no_side_effect_sample_single_particle(self):
        sigma_x = .5
        sigma_y = .5
        y_observed = 1

        def model(args):
            return {"y": st.norm(args['x'], sigma_y).rvs()}

        models = [model, model]
        models = list(map(SimpleModel, models))
        model_prior = RV("randint", 0, 2)
        nr_particles = 400
        mu_x_1, mu_x_2 = 0, 1
        parameter_given_model_prior_distribution = [Distribution(x=RV("norm", mu_x_1, sigma_x)),
                                                    Distribution(x=RV("norm", mu_x_2, sigma_x))]
        parameter_perturbation_kernels = [lambda t, stat: Kernel(stat['cov']) for _ in range(2)]
        abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.7),
                     parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                     PercentileDistanceFunction(measures_to_use=["y"]), MedianEpsilon(.2), nr_particles,
                     max_nr_allowed_sample_attempts_per_particle=2000)

        model_names = ["m1", "m2"]
        options = {'db_path': self.db}
        abc.set_data({"y": y_observed}, 0, {}, options, model_names)

        results = []
        for k in range(2):
            set_seeds()
            statistics = abc.history.get_statistics(-1)
            parameter_perturbation_kernels = abc._make_parameter_perturbation_kernels(statistics, 1)
            results.append(abc.map_wrapper.sample_single_particle(abc,parameter_perturbation_kernels, [4]*10, 0, 0, .2))

        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
