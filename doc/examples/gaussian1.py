# import sys
# sys.path.remove('')
# sys.path.insert(0, '../../../pyabc')

from pyabc import (ABCSMC, Distribution, RV,
                   PNormDistance, WeightedPNormDistance)
from pyabc.visualization import plot_kde_1d

import scipy
import tempfile
import os
import matplotlib.pyplot as pyplot


def model(p):
    return {'ss1': p['theta'] + 1 + 0.1*scipy.randn(),
            'ss2': 2 + scipy.randn()}
# ss1 is informative, ss2 is uninformative about theta


prior = Distribution(theta=RV('uniform', 0, 10))

distance = WeightedPNormDistance(p=2, adaptive=True)
distance = PNormDistance(2)
#sampler = SingleCoreSampler();
abc = ABCSMC(model, prior, distance, 100, lambda x: x, None, None, None, None)
db_path = ("sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db"))
observation1 = 4
observation2 = 2
# desirable: optimal theta=3
abc.new(db_path, {'ss1': observation1, 'ss2': observation2})

# run
history = abc.run(minimum_epsilon=.1, max_nr_populations=10)

# output
fig, ax = pyplot.subplots()
for t in range(history.max_t + 1):
    df, w = history.get_distribution(m=0, t=t)
    plot_kde_1d(df, w,
                xmin=0, xmax=10,
                x='theta', ax=ax,
                label="PDF t={}".format(t))
ax.axvline(observation1-1, color="k", linestyle="dashed")
ax.legend()
pyplot.show()

print("done test1")
