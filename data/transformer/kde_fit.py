#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:01:26 2016

@author: emmanuel
"""

import style
import pyabc
import scipy as sp
import pandas as pd

sm = style.make(output="test.npz",
                input="/home/emmanuel/abc/data/raw/toy:modes=2.db",
                wildcards=["4"])

history = pyabc.History("sqlite:///" + sm.input[0], 23, ["sdf"])
history.id = 1


transition = pyabc.MultivariateNormalTransition()


MAX_SIZE = 10


TX, TY = sp.meshgrid(sp.linspace(-MAX_SIZE, MAX_SIZE, 200),
                     sp.linspace(-MAX_SIZE, MAX_SIZE, 200))

TXY = sp.stack((TX, TY), axis=2)

fitted_kdes = []
ts = []
for t in range(1, history.max_t+1):
    df, w =  history.weighted_parameters_dataframe(t, 0)
    transition.fit(df, w)
    kdef = sp.array([[transition.pdf(pd.Series(theta, index=["theta1", "theta2"]))
                      for theta in pairs]
                      for pairs in TXY])
    fitted_kdes.append(kdef)
    ts.append(t)
    
import numpy as np
np.savez(sm.output[0][:-4], fitted_kdes=fitted_kdes, TX=TX, TY=TY)
