import pyabc
from pyabc.visualization import plot_kde_2d
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
import tempfile

db_path = "sqlite:///" + os.path.join(tempfile.gettempdir(), "test.db")

# MODEL

n_timepoints = 11
timepoints = sp.arange(n_timepoints)
init = sp.array([1, 0])


def f(x, t0, th0, th1):
    x0, x1 = x
    dx0 = - th0 * x0 + th1 * x1
    dx1 = th0 * x0 - th1 * x1
    return dx0, dx1


def model(p):
    sol = sp.integrate.odeint(f, init, timepoints, args=(p['th0'],p['th1']))
    return {'x1': sol[:,1]}


def distance(x, y):
    return sp.absolute(x['x1'] - y['x1']).sum()


class ArrayPNormDistance(pyabc.PNormDistance):

    def __init__(self):
        super().__init__(p=1)

    def initialize(self, t, sample_from_prior):
        sum_stats = []
        for sum_stat in sample_from_prior:
            sum_stats.append(normalize_sum_stat(sum_stat))
        super().initialize(t, sum_stats)

    def __call__(self, t, x, y):
        x = normalize_sum_stat(x)
        y = normalize_sum_stat(y)
        return super().__call__(t, x, y)


def normalize_sum_stat(x):
    x_flat = {}
    for key, value in x.items():
        for j in range(len(value)):
            x_flat[(key, j)] = value[j]
    return x_flat


# TRUE VALUES

th0_true, th1_true = sp.exp([-2.5, -2])
th_true = {'th0': th0_true, 'th1': th1_true}
data_true = model(th_true)

# MEASURED DATA


def f_data_meas():
    data_meas = np.zeros(n_timepoints)
    for _ in range(1000):
        data_meas += model(th_true)['x1'] + 0.01*np.random.rand(n_timepoints)
    data_meas /= 1000

    return {'x1': data_meas}


data_meas = f_data_meas()

# PERFORM ABC ANALYSIS

prior = pyabc.Distribution(th0=pyabc.RV('uniform', 0, 1),
                           th1=pyabc.RV('uniform', 0, 1))

distance = ArrayPNormDistance()
pop_size = 50
transition = pyabc.LocalTransition(k_fraction=.3)
eps = pyabc.MedianEpsilon(median_multiplier=.7)

abc = pyabc.ABCSMC(models=model,
                   parameter_priors=prior,
                   distance_function=distance,
                   population_size=pop_size,
                   transitions=transition,
                   eps=eps)

abc.new(db_path, data_meas)

h = abc.run(minimum_epsilon=0, max_nr_populations=5)

# PLOT

t = h.max_t

ax = plot_kde_2d(*h.get_distribution(m=0, t=t),
                 'th0', 'th1',
                 xmin=0, xmax=1, numx=300,
                 ymin=0, ymax=1, numy=300)
ax.scatter([th0_true], [th1_true],
           color='C1',
           label='$\Theta$ true = {:.3f}, {:.3f}'.format(th0_true, th1_true))
ax.set_title("Posterior t={}".format(t))
ax.legend()
plt.show()