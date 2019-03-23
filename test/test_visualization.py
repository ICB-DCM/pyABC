import pyabc
import tempfile
import pytest
import os
import numpy as np


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
for _ in range(n_history):
    abc = pyabc.ABCSMC(model, prior, distance)
    abc.new(db_path, observation)
    abc.run(minimum_epsilon=.1, max_nr_populations=4)


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
