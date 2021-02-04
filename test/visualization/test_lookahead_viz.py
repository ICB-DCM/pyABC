import pyabc
import tempfile
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# create and run some model


def model(p):
    return {'ss0': p['p0'] + 0.1 * np.random.uniform(),
            'ss1': p['p1'] + 0.1 * np.random.uniform()}


p_true = {'p0': 3, 'p1': 4}
observation = {'ss0': p_true['p0'], 'ss1': p_true['p1']}
limits = {'p0': (0, 5), 'p1': (1, 8)}
prior = pyabc.Distribution(**{
    key: pyabc.RV('uniform', limits[key][0], limits[key][1] - limits[key][0])
    for key in p_true.keys()})

db_path = "sqlite:///" + tempfile.mkstemp(suffix='.db')[1]
sampler_file = tempfile.mkstemp(suffix='.csv')[1]

distance = pyabc.PNormDistance(p=2)
sampler = pyabc.sampler.RedisEvalParallelSamplerServerStarter(
    look_ahead=True, look_ahead_delay_evaluation=True, log_file=sampler_file)

abc = pyabc.ABCSMC(model, prior, distance, population_size=20, sampler=sampler)
abc.new(db_path, observation)
history = abc.run(minimum_epsilon=.1, max_nr_populations=5)

sampler.shutdown()

sampler_df = pd.read_csv(sampler_file, sep=',')


def teardown_module():
    """Tear down module. Called after all tests here."""
    os.remove(db_path[len("sqlite:///"):])
    os.remove(sampler_file)


def test_lookahead_evaluations():
    """Test `pyabc.visualization.plot_lookahead_evaluations`"""
    for relative, fill in itertools.product([True, False], [True, False]):
        pyabc.visualization.plot_lookahead_evaluations(
            sampler_df, relative=relative, fill=fill, size=(5, 5))
    plt.close()


def test_lookahead_final_acceptance_fractions():
    """Test `pyabc.visualization.plot_lookahead_final_acceptance_fractions`"""
    for relative, fill in itertools.product([True, False], [True, False]):
        pyabc.visualization.plot_lookahead_final_acceptance_fractions(
            sampler_df, history, relative=relative, fill=fill, size=(5, 5))
    plt.close()


def test_lookahead_acceptance_rates():
    """Test `pyabc.visualization.plot_lookahead_acceptance_rates`"""
    pyabc.visualization.plot_lookahead_acceptance_rates(
        sampler_df, size=(5, 5))
    plt.close()
