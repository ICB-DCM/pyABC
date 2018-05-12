import scipy as sp
import pyabc
import pyabc.visualization
import matplotlib.pyplot as plt
import os
import tempfile
import logging


# for debugging
df_logger = logging.getLogger('DistanceFunction')
df_logger.setLevel(logging.DEBUG)


# model definition
def model(p):
    return {'ss1': p['theta'] + 1 + 0.1*sp.randn(),
            'ss2': 2 + 2*sp.randn()}


# true model parameter
theta_true = 3

# observed summary statistics
observation = {'ss1': theta_true + 1, 'ss2': 8}

# prior distribution
prior = pyabc.Distribution(theta=pyabc.RV('uniform', 0, 10))


# visualization
def visualize(history):
    fig, ax = plt.subplots()
    for t in range(history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        pyabc.visualization.plot_kde_1d(df, w, xmin=0, xmax=10,
                                        x='theta', ax=ax,
                                        label="PDF t={}".format(t))
    ax.axvline(theta_true, color='k', linestyle='dashed')
    ax.legend()
    plt.show()


# # NON-ADAPTIVE
# distance = pyabc.PNormDistance(p=2)
# abc = pyabc.ABCSMC(model, prior, distance)
db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), "tmp.db")
# abc.new(db_path, observation)
# h1 = abc.run(minimum_epsilon=0, max_nr_populations=8)
# visualize(h1)
#
# # ADAPTIVE
# distance = pyabc.AdaptivePNormDistance(p=2, adaptive=True)
# abc = pyabc.ABCSMC(model, prior, distance)
# abc.new(db_path, observation)
# h2 = abc.run(minimum_epsilon=0, max_nr_populations=8)
# visualize(h2)

# CENTERED
distance = pyabc.AdaptivePNormDistance(
    p=2,
    adaptive=True,
    scale_type=pyabc.AdaptivePNormDistance.SCALE_TYPE_C_MAD)
abc = pyabc.ABCSMC(model, prior, distance)
abc.new(db_path, observation)
h3 = abc.run(minimum_epsilon=0, max_nr_populations=8)
visualize(h3)


