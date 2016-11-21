#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 10:52:38 2016

@author: emmanuel
"""

# coding: utf-8

from style import make
import matplotlib.pyplot as plt
import scipy as sp
import pyabc
import parallel


#%%

sm = make(output="testdb.db", wildcards=["2"])
nr_modes = int(sm.wildcards[0])
print("Nr modes", nr_modes)

cov = sp.array([[.5, 0],
                [0, .5]])

data = sp.array([1, 1])


def square_nr_modes(theta):
    if nr_modes == 1:
        return theta
    if nr_modes == 4:
        return theta**2
    if nr_modes == 2:
        theta = theta.copy()
        theta[...,0] *= theta[...,0]
        return theta
    raise Exception("Nr modes {} not supported".format(nr_modes))

def p(x, theta):
    return 1 / sp.pi * sp.exp(-((x-square_nr_modes(theta))**2).sum(axis=-1))


MAX_SIZE = 10


TX, TY = sp.meshgrid(sp.linspace(-MAX_SIZE, MAX_SIZE, 200),
                     sp.linspace(-MAX_SIZE, MAX_SIZE, 200))


TXY = sp.stack((TX, TY), axis=2)


density = p(TXY, sp.array([0, 0]))

#%%
plt.pcolor(TX, TY, density);
plt.gca().set_aspect("equal")
#plt.savefig("/home/emmanuel/tmp/test.pdf", bbox_inches="tight")
plt.show()
#%%
def prior(theta):
    return 1/20**2 * ((-MAX_SIZE < theta) & (theta < MAX_SIZE)).all(axis=-1).astype(float)



def posterior(theta, data):
    return p(data, theta) * prior(theta)


plt.pcolor(TX, TY, posterior(TXY, data));
plt.gca().set_aspect("equal")
plt.colorbar()
plt.title("Posterior")
plt.show()



def abc_model(args):
    if nr_modes == 4:
        power_1, power_2 = 2, 2
    elif nr_modes == 2:
        power_1, power_2 = 2, 1
    elif nr_modes == 1:
        power_1, power_2 = 1, 1
    else:
        raise Exception("Invalid nr modes")
    theta_squared = sp.array([args.theta1**power_1, args.theta2**power_2])
    sample = sp.random.multivariate_normal(theta_squared, cov)
    return {"x": sample[0], "y": sample[1]}


class ABCPrior:
    def pdf(self, x):
        return prior(sp.array([x.theta1, x.theta2]))
    
    def rvs(self):
        sample = sp.rand(2) * 2 * MAX_SIZE - MAX_SIZE
        return pyabc.Parameter({"theta1": sample[0], "theta2": sample[1]})



model_prior = pyabc.RV("randint", 0, 1)
population_size = pyabc.AdaptivePopulationStrategy(500, 20,
                                                   max_population_size=10000)



mapper = parallel.SGE().map if parallel.sge_available() else map
abc = pyabc.ABCSMC([pyabc.SimpleModel(abc_model)],
                    model_prior,
                    pyabc.ModelPerturbationKernel(1, probability_to_stay=.8),
                    [ABCPrior()],
                    [pyabc.MultivariateNormalTransition()],
                    pyabc.PercentileDistanceFunction(measures_to_use=["x", "y"]),
                    pyabc.MedianEpsilon(),
                    population_size,
                    sampler=parallel.sampler.MappingSampler(map=mapper))
abc.stop_if_only_single_model_alive = False




options = {'db_path': "sqlite:///" + sm.output[0]}
abc.set_data({"x": 1, "y": 1}, 0, {}, options)




history = abc.run(.01)



#%%
points_theta, weights_theta =  history.weighted_parameters_dataframe(None, 0)
hist, xedges, yedges = sp.histogram2d(points_theta.theta1,
                                      points_theta.theta2,
                                      weights=weights_theta, bins=30);
xedges_mesh, yedges_mesh = sp.meshgrid(xedges[:-1], yedges[:-1])
plt.pcolor(xedges_mesh, yedges_mesh, hist)

