import pyabc
import tempfile
import pytest
import os
import numpy as np
from pyabc.visualization import data_plot

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

db_path = "sqlite:///" \
    + os.path.join(tempfile.gettempdir(), "test_visualize.db")


distance = pyabc.PNormDistance(p=2)
n_history = 2
sampler = pyabc.sampler.MulticoreEvalParallelSampler(n_procs=2)

for _ in range(n_history):
    abc = pyabc.ABCSMC(model, prior, distance, 20, sampler=sampler)
    abc.new(db_path, observation)
    abc.run(minimum_epsilon=.1, max_nr_populations=3)


histories = []
labels = []
for j in range(n_history):
    history = pyabc.History(db_path)
    history.id = j + 1
    histories.append(history)
    labels.append("Some run " + str(j))


def test_epsilons():
    pyabc.visualization.plot_sample_numbers(histories)
    pyabc.visualization.plot_sample_numbers(histories, labels)
    with pytest.raises(ValueError):
        pyabc.visualization.plot_sample_numbers(histories, [labels[0]])


def test_sample_numbers():
    pyabc.visualization.plot_sample_numbers(histories, labels, rotation=90)


def test_effective_sample_sizes():
    pyabc.visualization.plot_effective_sample_sizes(
        histories, labels, rotation=45)


def test_histograms():
    pyabc.visualization.plot_histogram_1d(histories[0], 'p0', bins=20)
    pyabc.visualization.plot_histogram_2d(histories[0], 'p0', 'p1')
    pyabc.visualization.plot_histogram_matrix(histories[0], bins=1000)


def test_kdes():
    df, w = histories[0].get_distribution(m=0, t=None)
    pyabc.visualization.plot_kde_1d(
        df, w, x='p0',
        xmin=limits['p0'][0], xmax=limits['p0'][1],
        label="PDF")
    pyabc.visualization.plot_kde_2d(df, w, x='p0', y='p1')
    pyabc.visualization.plot_kde_matrix(df, w)


def test_credible_intervals():
    pyabc.visualization.plot_credible_intervals(histories[0])
    pyabc.visualization.plot_credible_intervals(
        histories[0], levels=[0.2, 0.5, 0.9])
    pyabc.visualization.plot_credible_intervals_for_time(
        histories, levels=[0.5, 0.99],
        show_kde_max_1d=True, show_kde_max=True, show_mean=True,
        refvals=p_true)


def test_model_probabilities():
    pyabc.visualization.plot_model_probabilities(histories[0])


def test_data_plot():
    obs_dict = {1: 0.7}
    sim_dict = {1: 6.5}
    data_plot.plot_data(obs_dict, sim_dict)
    for i in range(18):
        obs_dict[i] = i + 1
        sim_dict[i] = i + 2
    data_plot.plot_data(obs_dict, sim_dict)
