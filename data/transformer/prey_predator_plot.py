#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:37:53 2016

@author: emmanuel


To adjust the number of exapmles, adjust the Sankefile
"""
import style
from style import make
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

sm = make()

simulations = list(map(sp.load, sm.input))

simulations_grouped = {}
for sim in simulations:
    simulations_grouped.setdefault(int(sim["model_nr"]), []).append(sim)
    
max_t = .1


max_nr_seeds = max(len(simulations_grouped[1]), len(simulations_grouped[2]))

fig, axes = plt.subplots(2, max_nr_seeds, sharex=True, sharey=True)
fig.set_size_inches((style.size.m[0]*len(simulations_grouped[1]), style.size.m[1]*2))
for model, row  in zip([1, 2], axes):
    row[0].set_ylabel("Model " + str(model) + "\n" + "Species")
    for ax, sim in zip(row, simulations_grouped[model]):
        t, X = sim["t"], sim["X"]
        ax.step(t, X)
        ax.set_xlim(0, max_t)
        ax.set_xticks([0, max_t])
        max_y = 45
        ax.set_ylim(0, max_y)
        ax.set_yticks([0, max_y])

for ax in row:
    ax.set_xlabel(style.name.t, labelpad=style.labelpad.reduced)
        
ax.legend(["X", "Y"], loc="center left", bbox_to_anchor=(1, .5))
sns.despine(fig)
fig.tight_layout()
fig.save(sm.output[0])
