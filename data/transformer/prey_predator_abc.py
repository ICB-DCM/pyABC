#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 14:30:24 2016

@author: emmanuel
"""

from style import make
import matplotlib.pyplot as plt
import scipy as sp
import pyabc
import parallel
import os
from data.producer.prey_predator import Model1, Model2

#%%

sm = make(output=os.path.expanduser("~/tmp/test_preyabc.db"))

model_prior = pyabc.RV("randint", 0, 2)
population_strategy = pyabc.AdaptivePopulationStrategy(500, 20,
                                                   max_population_size=10000)

rate_prior = pyabc.Distribution(rate=pyabc.RV("uniform", 0, 100))


def model_1(pars):
    rate = pars.rate
    arr = sp.rand(4)
    return arr

    
def model_2(pars):
    rate = pars.rate
    arr = sp.rand(4)
    return arr


def distance(x, y):
        return ((x - y)**2).sum()
    

mapper = parallel.SGE().map if parallel.sge_available() else map
abc = pyabc.ABCSMC([pyabc.SimpleModel(model_1),
                    pyabc.SimpleModel(model_2)],
                    model_prior,
                    pyabc.ModelPerturbationKernel(2, probability_to_stay=.8),
                    [rate_prior, rate_prior],
                    [pyabc.MultivariateNormalTransition(),
                     pyabc.MultivariateNormalTransition()],
                    distance,
                    pyabc.MedianEpsilon(),
                    population_strategy,
                    sampler=parallel.sampler.MappingSampler(map=mapper))
abc.stop_if_only_single_model_alive = False


options = {'db_path': "sqlite:///" + sm.output[0]}
abc.set_data(sp.rand(4), 0, {}, options)
history = abc.run(.01)
