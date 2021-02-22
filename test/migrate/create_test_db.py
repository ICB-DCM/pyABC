"""Create a small test database to be used in migration tests."""

import numpy as np
import os
import tempfile

import pyabc


def model(p):
    return {'ss0': p['p0'] + 0.1 * np.random.uniform(),
            'ss1': p['p1'] + 0.1 * np.random.uniform()}


p_true = {'p0': 3, 'p1': 4}
limits = {'p0': (0, 5), 'p1': (1, 8)}

observation = {'ss0': p_true['p0'], 'ss1': p_true['p1']}

prior = pyabc.Distribution(**{
    key: pyabc.RV('uniform', limits[key][0],
                  limits[key][1] - limits[key][0])
    for key in p_true.keys()})

distance = pyabc.PNormDistance(p=2)

abc = pyabc.ABCSMC(model, prior, distance, population_size=10)
db_file = os.path.join(tempfile.gettempdir(), 'pyabc_test_migrate.db')
abc.new("sqlite:///" + db_file, observation)
abc.run(minimum_epsilon=.1, max_nr_populations=3)
