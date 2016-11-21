#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:29:11 2016

@author: emmanuel
"""
import style
import scipy as sp
import matplotlib.pyplot as plt


sm = style.make(output="test.pdf",
                input="test.npz",
                wildcards=["4"])

loaded = sp.load(sm.input[0])

fitted_kdes = loaded["fitted_kdes"]
TX = loaded["TX"]
TY = loaded["TY"]


#%%
fig, axes = plt.subplots(fitted_kdes.shape[0])
for t, (kdefit, ax) in enumerate(zip(fitted_kdes, axes)):
    mappable = ax.pcolor(TX, TY, kdefit);
    
    ax.set_aspect("equal")
    ax.set_title("t={}".format(t+1))
    fig.colorbar(mappable)
    fig.show()
    fig.save(sm.output[0])
