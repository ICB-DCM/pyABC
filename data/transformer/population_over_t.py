#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:29:11 2016

@author: emmanuel
"""
import style
import pyabc
import scipy as sp
import matplotlib.pyplot as plt
import pandas as pd

sm = style.make(output="test.pdf",
                input="/home/emmanuel/abc/data/raw/toy:modes=2.db",
                wildcards=["4"])

history = pyabc.History("sqlite:///" + sm.input[0], 23, ["sdf"])
history.id = 1


transition = pyabc.MultivariateNormalTransition()


MAX_SIZE = 10


TX, TY = sp.meshgrid(sp.linspace(-MAX_SIZE, MAX_SIZE, 200),
                     sp.linspace(-MAX_SIZE, MAX_SIZE, 200))

TXY = sp.stack((TX, TY), axis=2)


for t in range(history.max_t):
    df, w =  history.weighted_parameters_dataframe(t, 0)
    transition.fit(df, w)
    kdef = sp.array([[transition.pdf(pd.Series(theta, index=["theta1", "theta2"]))
                      for theta in pairs]
                      for pairs in TXY])
    fig, ax = plt.subplots()
    ax.pcolor(TX, TY, kdef);
    
    ax.set_aspect("equal")
    ax.set_title("t={}".format(t))
    fig.colorbar()
