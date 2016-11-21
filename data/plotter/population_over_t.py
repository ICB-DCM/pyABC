#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:29:11 2016

@author: emmanuel
"""
import style
import scipy as sp
import matplotlib.pyplot as plt
import math
import seaborn as sns


sm = style.make(output="test.pdf",
                input="/home/emmanuel/abc/data"
                "/processed/toy-fitted-kde:modes=1.npz",
                wildcards=["1"])

loaded = sp.load(sm.input[0])

fitted_kdes = loaded["fitted_kdes"]
nr_kdes = fitted_kdes.shape[0]
TX = loaded["TX"]
TY = loaded["TY"]

n_rows = int(math.sqrt(nr_kdes))
n_cols = math.ceil(nr_kdes / n_rows)

#%%
fig, axes = plt.subplots(n_rows, n_cols, sharex=True, sharey=True)
fig.set_size_inches((1.2*n_cols, .9*n_rows))

for ax in axes.flatten():
    ax.axis("off")

for t, (kdefit, ax) in enumerate(zip(fitted_kdes, axes.flatten())):
    ax.axis("on")
    mappable = ax.pcolor(TX, TY, kdefit,
                         vmax=fitted_kdes.max(), vmin=fitted_kdes.min(),
                         clip_on=False);
    
    
    ax.axis("tight")
    #ax.set_aspect("equal")
    ax.set_title("$t={}$".format(t+1), y=.89)
    ax.set_xlim(TX.min(), TX.max())
    ax.set_ylim(TY.min(), TY.max())
    style.middle_ticks_minor(ax)
    
     
sns.despine(fig)    

cax = fig.add_axes([.95, .35, .05, .3])

import matplotlib as mpl
ticker = mpl.ticker.MaxNLocator(4)
cb = fig.colorbar(mappable, ticks=ticker, cax=cax)
cb.outline.set_linewidth(0)
cb.set_label(style.name.pdf)


fig.show()
fig.save(sm.output[0])
