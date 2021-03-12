import pyabc
import tempfile
import os
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


db_path = "sqlite:///" + tempfile.mkstemp(suffix='.db')[1]
histories = []
labels = []


def model(p):
    return {'ss0': p['p0'] + 0.1 * np.random.uniform(),
            'ss1': p['p1'] + 0.1 * np.random.uniform()}


p_true = {'p0': 3, 'p1': 4}
limits = {'p0': (0, 5), 'p1': (1, 8)}


def setup_module():
    """Set up module. Called before all tests here."""
    # create and run some model
    observation = {'ss0': p_true['p0'], 'ss1': p_true['p1']}

    prior = pyabc.Distribution(**{
        key: pyabc.RV('uniform', limits[key][0],
                      limits[key][1] - limits[key][0])
        for key in p_true.keys()})

    distance = pyabc.PNormDistance(p=2)
    n_history = 2
    sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=2)

    for _ in range(n_history):
        abc = pyabc.ABCSMC(model, prior, distance, population_size=100,
                           sampler=sampler)
        abc.new(db_path, observation)
        abc.run(minimum_epsilon=.1, max_nr_populations=3)

    for j in range(n_history):
        history = pyabc.History(db_path)
        history.id = j + 1
        histories.append(history)
        labels.append("Some run " + str(j))


def teardown_module():
    """Tear down module. Called after all tests here."""
    os.remove(db_path[len("sqlite:///"):])


def test_set_figure_params():
    """Test setting figure parameters globally."""
    pyabc.settings.set_figure_params('pyabc')
    # plot something
    pyabc.visualization.plot_kde_matrix_highlevel(
        histories[0], refval=p_true)
    pyabc.settings.set_figure_params('default')
    # plot something else
    pyabc.visualization.plot_walltime(histories, labels)
    with pytest.raises(ValueError):
        pyabc.settings.set_figure_params('pypesto')
    plt.close()


def test_epsilons():
    """Test `pyabc.visualization.plot_epsilons`"""
    pyabc.visualization.plot_epsilons(histories, labels)
    plt.close()


def test_sample_numbers():
    """Test `pyabc.visualization.plot_sample_numbers`"""
    pyabc.visualization.plot_sample_numbers(
        histories, rotation=43, size=(5, 5))
    _, ax = plt.subplots()
    pyabc.visualization.plot_sample_numbers(histories, labels, ax=ax)
    pyabc.visualization.plot_sample_numbers(histories, labels[0])
    plt.close()


def test_sample_numbers_trajectory():
    """Test `pyabc.visualization.plot_sample_numbers_trajectory`"""
    pyabc.visualization.plot_sample_numbers_trajectory(
        histories, labels, yscale='log', rotation=90)
    _, ax = plt.subplots()
    pyabc.visualization.plot_sample_numbers_trajectory(
        histories, labels, yscale='log10', size=(8, 8), ax=ax)
    plt.close()


def test_acceptance_rates_trajectory():
    """Test `pyabc.visualization.plot_acceptance_rates_trajectory`"""
    pyabc.visualization.plot_acceptance_rates_trajectory(
        histories, labels, yscale='log')
    _, ax = plt.subplots()
    pyabc.visualization.plot_acceptance_rates_trajectory(
        histories, labels, yscale='log10', size=(10, 5), ax=ax,
        normalize_by_ess=True)
    pyabc.visualization.plot_acceptance_rates_trajectory(
        histories, labels, yscale='log10', size=(10, 5), ax=ax,
        normalize_by_ess=False)
    plt.close()


def test_total_sample_numbers():
    """Test `pyabc.visualization.plot_total_sample_numbers`"""
    pyabc.visualization.plot_total_sample_numbers(histories)
    pyabc.visualization.plot_total_sample_numbers(
        histories, labels, yscale='log', size=(10, 5))
    _, ax = plt.subplots()
    pyabc.visualization.plot_total_sample_numbers(
        histories, rotation=75, yscale='log10', ax=ax)
    plt.close()


def test_effective_sample_sizes():
    """Test `pyabc.visualization.plot_effective_sample_numbers`"""
    pyabc.visualization.plot_effective_sample_sizes(
        histories, labels, rotation=45, relative=True)
    plt.close()


def test_histograms():
    """Test `pyabc.visualization.plot_histogram_1d/2d/matrix`"""
    # 1d
    pyabc.visualization.plot_histogram_1d(
        histories[0], 'p0', bins=20,
        xmin=limits['p0'][0], xmax=limits['p0'][1], size=(5, 5), refval=p_true)
    # 2d
    pyabc.visualization.plot_histogram_2d(histories[0], 'p0', 'p1')
    pyabc.visualization.plot_histogram_2d(
        histories[0], 'p0', 'p1', xmin=limits['p0'][0], xmax=limits['p0'][1],
        ymin=limits['p1'][0], ymax=limits['p1'][1], size=(5, 6), refval=p_true)
    # matrix
    pyabc.visualization.plot_histogram_matrix(
        histories[0], bins=1000, size=(6, 7), refval=p_true)
    plt.close()


def test_kdes():
    """Test `pyabc.visualization.plot_kde_1d/2d/matrix` and highlevel
    versions."""
    history = histories[0]
    df, w = history.get_distribution(m=0, t=None)
    pyabc.visualization.plot_kde_1d(
        df, w, x='p0',
        xmin=limits['p0'][0], xmax=limits['p0'][1],
        label="PDF")
    pyabc.visualization.plot_kde_2d(df, w, x='p0', y='p1')
    pyabc.visualization.plot_kde_matrix(df, w)

    # also use the highlevel interfaces
    pyabc.visualization.plot_kde_1d_highlevel(
        history, x='p0', size=(4, 5), refval=p_true)
    pyabc.visualization.plot_kde_2d_highlevel(
        history, x='p0', y='p1', size=(7, 5), refval=p_true)
    pyabc.visualization.plot_kde_matrix_highlevel(
        history, height=27.43, refval=p_true)
    plt.close()


def test_credible_intervals():
    """Test `pyabc.visualization.plot_credible_intervals` and
    `pyabc.visualization.plot_credible_intervals_for_time`"""
    pyabc.visualization.plot_credible_intervals(histories[0])
    pyabc.visualization.plot_credible_intervals(
        histories[0], levels=[0.2, 0.5, 0.9],
        show_kde_max_1d=True, show_kde_max=True, show_mean=True,
        refval=p_true)
    pyabc.visualization.plot_credible_intervals_for_time(
        histories, levels=[0.5, 0.99],
        show_kde_max_1d=True, show_kde_max=True, show_mean=True,
        refvals=p_true)
    plt.close()


def test_model_probabilities():
    """Test `pyabc.visualization.plot_model_probabilities`"""
    pyabc.visualization.plot_model_probabilities(histories[0])
    plt.close()


def test_data_callback():
    """Test `pyabc.visualization.plot_data_callback`"""
    def plot_data(sum_stat, weight, ax, **kwargs):
        ax.plot(sum_stat['ss0'], alpha=weight, **kwargs)

    def plot_data_aggregated(sum_stats, weights, ax, **kwargs):
        data = np.array([sum_stat['ss0'] for sum_stat in sum_stats])
        weights = np.array(weights).reshape((-1, 1))
        mean = (data * weights).sum(axis=0)
        plot_data({'ss0': mean}, 1.0, ax)

    pyabc.visualization.plot_data_callback(
        histories[0], plot_data, plot_data_aggregated)
    plt.close()


def test_data_default():
    """Test `pyabc.visualization.plot_data_default`"""
    obs_dict = {1: 0.7, 2: np.array([43, 423, 5.5]),
                3: pd.DataFrame({'a': [1, 2], 'b': [4, 6]})}
    sim_dict = {1: 6.5, 2: np.array([32, 5, 6]),
                3: pd.DataFrame({'a': [1.55, -0.1], 'b': [54, 6]})}
    pyabc.visualization.plot_data_default(obs_dict, sim_dict)
    for i in range(5):
        obs_dict[i] = i + 1
        sim_dict[i] = i + 2
    pyabc.visualization.plot_data_default(obs_dict, sim_dict)
    plt.close()


def test_total_walltime():
    """Test `pyabc.visualization.plot_total_walltime`"""
    pyabc.visualization.plot_total_walltime(
        histories, labels, rotation=45, unit='m', size=(5, 5))
    with pytest.raises(AssertionError):
        pyabc.visualization.plot_total_walltime(histories, unit='min')
    plt.close()


def test_walltime():
    """Test `pyabc.visualization.plot_walltime`"""
    pyabc.visualization.plot_walltime(
        histories, labels, rotation=45, unit='m', size=(5, 5))
    with pytest.raises(AssertionError):
        pyabc.visualization.plot_walltime(histories, unit='min')
    plt.close()


def test_eps_walltime():
    """Test `pyabc.visualization.plot_eps_walltime`"""
    pyabc.visualization.plot_eps_walltime(
        histories, labels, unit='m', size=(5, 5), yscale='log')
    with pytest.raises(AssertionError):
        pyabc.visualization.plot_eps_walltime(histories, unit='min')
    plt.close()
