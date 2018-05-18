import scipy
import scipy.stats
import numpy


c = 0.8


def gk(pars):
    # extract parameters
    A = pars['A']
    B = pars['B']
    g = pars['g']
    k = pars['k']

    # sample from normal distribution
    z = numpy.random.normal(0, 1)

    # to sample from gk distribution
    e = numpy.exp(-g*z)
    return A + B * (1 + c * (1-e) / (1+e)) * (1 + z**2)**k * z


theta0 = {'A': 3.0,
          'B': 1.0,
          'g': 1.5,
          'k': 0.5}


# observed sum stats

# prior

# distance

# abc