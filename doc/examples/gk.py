import numpy
import pyabc.visualization
import os
import tempfile
import matplotlib.pyplot as plt


# model


def gk(pars):
    # extract parameters
    a = pars['A']
    b = pars['B']
    g = pars['g']
    k = pars['k']
    c = 0.8

    # sample from normal distribution
    z = numpy.random.normal(0, 1)

    # to sample from gk distribution
    return z_to_gk(a, b, g, k, c, z)


def z_to_gk(a, b, g, k, c, z):
    e = numpy.exp(-g*z)
    return a + b * (1 + c * (1-e) / (1+e)) * (1 + z**2)**k * z


n_dataset = 10000
order_statistics_indices = [1250 * j for j in range(1, n_dataset//1250)]


def data_gk(pars):
    data = [gk(pars) for _ in range(0,n_dataset)]
    return data


# sum stats


def ordered_statistics_gk(data: list):
    data.sort()
    order_statistics = {j: data[j] for j in order_statistics_indices}
    return order_statistics


# observations

# true parameters
theta0 = {'A': 3.0,
          'B': 1.0,
          'g': 1.5,
          'k': 0.5}

# observed data
obs_data = data_gk(theta0)

# observed summary statistics
obs_sum_stats = ordered_statistics_gk(obs_data)

# prior
prior = pyabc.Distribution(**{key: pyabc.RV('uniform', 0, 10)
                              for key in theta0})

# distance
distance = pyabc.AdaptivePNormDistance()

# acceptor
acceptor = pyabc.accept_use_complete_history

# abc

db_name = "sqlite:///" + os.path.join(tempfile.gettempdir(), "tmp.db")

abc = pyabc.ABCSMC(models=pyabc.SimpleModel(data_gk),
                   parameter_priors=prior,
                   distance_function=distance,
                   summary_statistics=ordered_statistics_gk,
                   population_size=100,
                   acceptor=acceptor)
abc.new(db=db_name, observed_sum_stat=obs_sum_stats)
abc.run(minimum_epsilon=0, max_nr_populations=10)

# visualization

df, w = abc.history.get_distribution(m=0)
pyabc.visualization.plot_kde_matrix(df, w, limits={key: (0,10)
                                                   for key in theta0})
plt.show()