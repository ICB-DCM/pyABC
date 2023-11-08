"""Test visualization using plotly as backend."""

import os
import tempfile

import numpy as np
import plotly.graph_objects as go
import pytest

import pyabc

db_path = "sqlite:///" + tempfile.mkstemp(suffix='.db')[1]
log_files = []
histories = []
labels = []


def model(p):
    return {
        'ss0': p['p0'] + 0.1 * np.random.uniform(),
        'ss1': p['p1'] + 0.1 * np.random.uniform(),
    }


p_true = {'p0': 3, 'p1': 4}
limits = {'p0': (0, 5), 'p1': (1, 8)}


def setup_module():
    """Set up module. Called before all tests here."""
    # create and run some model
    observation = {'ss0': p_true['p0'], 'ss1': p_true['p1']}

    prior = pyabc.Distribution(
        **{
            key: pyabc.RV(
                'uniform', limits[key][0], limits[key][1] - limits[key][0]
            )
            for key in p_true.keys()
        }
    )

    n_history = 2
    sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=2)

    for _ in range(n_history):
        log_file = tempfile.mkstemp(suffix=".json")[1]
        log_files.append(log_file)
        distance = pyabc.AdaptivePNormDistance(p=2, scale_log_file=log_file)

        abc = pyabc.ABCSMC(
            model, prior, distance, population_size=100, sampler=sampler
        )
        abc.new(db_path, observation)
        abc.run(minimum_epsilon=0.1, max_nr_populations=3)

    for j in range(n_history):
        history = pyabc.History(db_path)
        history.id = j + 1
        histories.append(history)
        labels.append("Some run " + str(j))


def teardown_module():
    """Tear down module. Called after all tests here."""
    os.remove(db_path[len("sqlite:///") :])
    for log_file in log_files:
        os.remove(log_file)


def test_epsilons():
    """Test `pyabc.visualization.plot_epsilons`"""
    pyabc.visualization.plot_epsilons_plotly(histories, labels)


def test_sample_numbers():
    """Test `pyabc.visualization.plot_sample_numbers`"""
    pyabc.visualization.plot_sample_numbers_plotly(
        histories, rotation=43, size=(500, 500)
    )
    fig = go.Figure()
    pyabc.visualization.plot_sample_numbers_plotly(histories, labels, fig=fig)
    pyabc.visualization.plot_sample_numbers_plotly(histories, labels[0])


def test_sample_numbers_trajectory():
    """Test `pyabc.visualization.plot_sample_numbers_trajectory`"""
    pyabc.visualization.plot_sample_numbers_trajectory_plotly(
        histories, labels, yscale='log'
    )
    fig = go.Figure()
    pyabc.visualization.plot_sample_numbers_trajectory_plotly(
        histories, labels, yscale='log10', size=(800, 800), fig=fig
    )


def test_acceptance_rates_trajectory():
    """Test `pyabc.visualization.plot_acceptance_rates_trajectory`"""
    pyabc.visualization.plot_acceptance_rates_trajectory_plotly(
        histories, labels, yscale='log'
    )
    fig = go.Figure()
    pyabc.visualization.plot_acceptance_rates_trajectory_plotly(
        histories,
        labels,
        yscale='log10',
        size=(1000, 500),
        fig=fig,
        normalize_by_ess=True,
    )
    pyabc.visualization.plot_acceptance_rates_trajectory_plotly(
        histories,
        labels,
        yscale='log10',
        size=(1000, 500),
        fig=fig,
        normalize_by_ess=False,
    )


def test_total_sample_numbers():
    """Test `pyabc.visualization.plot_total_sample_numbers`"""
    pyabc.visualization.plot_total_sample_numbers_plotly(histories)
    pyabc.visualization.plot_total_sample_numbers_plotly(
        histories, labels, yscale='log', size=(1000, 500)
    )
    fig = go.Figure()
    pyabc.visualization.plot_total_sample_numbers_plotly(
        histories, rotation=75, yscale='log10', fig=fig
    )


def test_effective_sample_sizes():
    """Test `pyabc.visualization.plot_effective_sample_numbers`"""
    pyabc.visualization.plot_effective_sample_sizes_plotly(
        histories, labels, rotation=45, relative=True
    )


def test_kdes():
    """Test `pyabc.visualization.plot_kde_1d/2d/matrix` and highlevel
    versions."""
    history = histories[0]
    df, w = history.get_distribution(m=0, t=None)
    pyabc.visualization.plot_kde_1d_plotly(
        df, w, x='p0', xmin=limits['p0'][0], xmax=limits['p0'][1]
    )
    pyabc.visualization.plot_kde_2d_plotly(df, w, x='p0', y='p1')
    pyabc.visualization.plot_kde_matrix_plotly(df, w)

    # also use the highlevel interfaces
    pyabc.visualization.plot_kde_1d_highlevel_plotly(
        history, x='p0', size=(400, 500), refval=p_true
    )
    pyabc.visualization.plot_kde_2d_highlevel_plotly(
        history, x='p0', y='p1', size=(700, 500), refval=p_true
    )
    pyabc.visualization.plot_kde_matrix_highlevel_plotly(
        history, height=27.43, refval=p_true
    )


def test_credible_intervals():
    """Test `pyabc.visualization.plot_credible_intervals` and
    `pyabc.visualization.plot_credible_intervals_for_time`"""
    pyabc.visualization.plot_credible_intervals_plotly(histories[0])
    pyabc.visualization.plot_credible_intervals_plotly(
        histories[0],
        levels=[0.2, 0.5, 0.9],
        refval=p_true,
    )


def test_model_probabilities():
    """Test `pyabc.visualization.plot_model_probabilities`"""
    pyabc.visualization.plot_model_probabilities_plotly(histories[0])


def test_total_walltime():
    """Test `pyabc.visualization.plot_total_walltime`"""
    pyabc.visualization.plot_total_walltime_plotly(
        histories,
        labels,
        rotation=45,
        unit='m',
        size=(500, 500),
    )
    with pytest.raises(AssertionError):
        pyabc.visualization.plot_total_walltime_plotly(histories, unit='min')


def test_walltime():
    """Test `pyabc.visualization.plot_walltime`"""
    pyabc.visualization.plot_walltime_plotly(
        histories, labels, rotation=45, unit='m', size=(500, 500)
    )
    with pytest.raises(AssertionError):
        pyabc.visualization.plot_walltime_plotly(histories, unit='min')


def test_eps_walltime():
    """Test `pyabc.visualization.plot_eps_walltime`"""
    for group_by_label in [True, False]:
        pyabc.visualization.plot_eps_walltime_plotly(
            histories,
            labels,
            unit='m',
            size=(500, 500),
            yscale='log',
            group_by_label=group_by_label,
        )
    with pytest.raises(AssertionError):
        pyabc.visualization.plot_eps_walltime_plotly(histories, unit='min')
