import unittest
from pyabc import (ABCSMC, RV, ModelPerturbationKernel, Distribution,
                   MedianEpsilon, MinMaxDistanceFunction, PercentileDistanceFunction, SimpleModel, Model, ModelResult,
                   MultivariateNormalTransition)
import random
import os
import scipy.stats as st
from scipy.special import gamma, binom
import scipy as sp
import scipy.interpolate
import multiprocessing
import tempfile
import parallel.sampler as ps
from pyabc.populationstrategy import ConstantPopulationStrategy, AdaptivePopulationStrategy

REMOVE_DB = False


def mean_and_std(values, weights):
    mean = (values * weights).sum()
    std = sp.sqrt(((values - mean)**2 * weights).sum())
    return mean, std


class AllInOneModel(Model):
    def summary_statistics(self, pars, sum_stats_calculator) -> ModelResult:
        return ModelResult(sum_stats={"result": 1})

    def accept(self, pars, sum_stats_calculator, distance_calculator, eps) -> ModelResult:
        return ModelResult(accepted=True)


class TestABC(unittest.TestCase):
    def setUp(self):
        self.db_file_location = os.path.join(tempfile.gettempdir(), "abc_unittest.db")
        self.db = "sqlite:///" + self.db_file_location
        self.clean_db()

    def clean_db(self):
        try:
            if REMOVE_DB:
                os.remove(self.db_file_location)
        except FileNotFoundError:
            pass

    def tearDown(self):
        self.clean_db()


class TestABCFast(TestABC):
    def test_all_in_one_model(self):
        models = [AllInOneModel() for _ in range(2)]
        model_prior = RV("randint", 0, 2)
        mp_pool = multiprocessing.Pool(4)
        mp_sampler = ps.MappingSampler(map=mp_pool.map)
        population_size = ConstantPopulationStrategy(800, 3)
        parameter_given_model_prior_distribution = [Distribution(theta=RV("beta", 1, 1)) for _ in range(2)]
        parameter_perturbation_kernels = [MultivariateNormalTransition() for _ in range(2)]
        abc = ABCSMC(models, model_prior, ModelPerturbationKernel(2, probability_to_stay=.8),
                     parameter_given_model_prior_distribution, parameter_perturbation_kernels,
                     MinMaxDistanceFunction(measures_to_use=["result"]), MedianEpsilon(.1), population_size,
                     sampler=mp_sampler)

        options = {'db_path': self.db}
        abc.set_data({"result": 2}, 0, {}, options)

        minimum_epsilon = .2
        history = abc.run(minimum_epsilon)
        mp = history.get_model_probabilities(history.max_t)
        # self.assertLess(abs(p1 - .5) + abs(p2 - .5), .08)
        self.assertLess(abs(mp.p[0] - .5) + abs(mp.p[1] - .5), .08*5) # Dennis


if __name__ == "__main__":
    unittest.main()
