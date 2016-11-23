#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 15:04:51 2016

@author: emmanuel
"""
import scipy as sp
import matplotlib.pyplot as plt
import style
import seaborn as sns
import time

def h(x, pre, c):
    return (x**pre).prod(1) * c


def fast_random_choice(weights):
    """
    this is at least for small arrays much faster
    than numpy.random.choice.
    For the Gillespie overall this brings for 3 reaction a speedup of a factor of 2
    """
    cs = 0
    u = sp.random.rand()
    for k in range(weights.size):
        cs += weights[k]
        if u <= cs:
            return k
    raise Exception("Random choice error {}".format(weights))

    
def gillespie(x, c, pre, post, max_t):
    """
    Parameters
    ----------
    
    x: 1D array, size nr_species
        initial numbers
    
    c: 1D array, size nr_reactions
        reaction rates
    
    pre: array nr_reactions x nr_species
        consumed    
    
    post: array nr_reactions x nr_species
        what is to be produced
    
    max_t: int
        up to where to simulate
    
    """
    t = 0
    t_store = [t]
    x_store = [x.copy()]
    S = post - pre

    while t < max_t:
        h_vec = h(x, pre, c)
        h0 = h_vec.sum()
        delta_t = sp.random.exponential(1 / h0)
        if not sp.isfinite(delta_t):
            break
        reaction = fast_random_choice_cython(h_vec/h0)
        #reaction = sp.random.choice(c.size, p=h_vec/h0)
        t = t + delta_t
        x = x + S[reaction]
        
        t_store.append(t)
        x_store.append(x)

    return sp.asarray(t_store), sp.asarray(x_store)
        
    
if __name__ == "__main__"    :
    x = sp.array([50, 100])   # molecule numbers
    c = sp.array([1, .005, .6])  # rate constants
    pre = sp.array([[1, 0],
                    [1, 1],
                    [0, 1]], dtype=int)
    post = sp.array([[2, 0],
                     [0, 2],
                     [0, 0]])
    max_t = 25    
        
    start = time.time()    
    t_store, x_store = gillespie(x, c, pre, post, max_t)    
    duration = time.time() - start    
    
    print(len(t_store), "steps", duration, "duration")
    
    fig, ax = plt.subplots()    
    ax.plot(sp.array(t_store), sp.asarray(x_store))
    ax.legend(["Prey", "Predator"], loc="center left", bbox_to_anchor=[1, .5])
    sns.despine(ax=ax)
    style.middle_ticks_minor(ax)