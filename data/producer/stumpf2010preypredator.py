#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 10:08:49 2016

@author: emmanuel
"""

import scipy as sp
from data.producer.gillespie import gillespie
import style
import seaborn as sns
import matplotlib.pyplot as plt


class Model1:
    name = "Model 1"
    x0 = sp.array([40, 3])   # molecule numbers
    c = sp.array([2.1])  # rate constants
    pre = sp.array([[1, 1]], dtype=int)
    post = sp.array([[0, 2]])
    max_t = .1

    
class Model2(Model1):
    name = "Model 2"
    c = sp.array([30])  # rate constants
    pre = sp.array([[1, 0]], dtype=int)
    post = sp.array([[0, 1]])
    
    
fig, axes = plt.subplots(2, 5, sharex=True, sharey=True)
fig.set_size_inches((style.size.m[0]*5, style.size.m[1]*2))
for model, row  in zip([Model1, Model2], axes):
    row[0].set_ylabel(model.name + "\n" + "Species")
    for ax in row:
        t, x = gillespie(model.x0, model.c, model.pre, model.post, model.max_t)
        ax.plot(t, x)
        ax.set_xlim(0, model.max_t)
        ax.set_xticks([0, model.max_t])
        max_y = 45
        ax.set_ylim(0, max_y)
        ax.set_yticks([0, max_y])

for ax in row:
    ax.set_xlabel(style.name.t, labelpad=style.labelpad.reduced)
        
ax.legend(["X", "Y"], loc="center left", bbox_to_anchor=(1, .5))
sns.despine(fig)
fig.tight_layout()
fig.show()