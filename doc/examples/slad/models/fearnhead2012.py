"""Models from Fearnhead, Prangle 2012, Constructing Summary Statistics for
ABC.
"""

import numpy as np
from typing import Callable

from pyabc import Distribution, RV
import pyabc
from .base import Problem, gk


class FearnheadGKProblem(Problem):

    def get_model(self) -> Callable:
        ixs = np.linspace(0, 1000, 102, dtype=int)[1:-1]

        def model(p):
            A, B, g, k = [p[key] for key in ['A', 'B', 'g', 'k']]
            c = 0.8
            vals = gk(A=A, B=B, c=c, g=g, k=k, n=10000)
            ordered = np.sort(vals)
            subset = ordered[ixs]
            return {'y': subset}

        return model

    def get_prior(self) -> Distribution:
        return Distribution(
            A=RV('uniform', 0, 10),
            B=RV('uniform', 0, 10),
            g=RV('uniform', 0, 10),
            k=RV('uniform', 0, 10))

    def get_prior_bounds(self) -> dict:
        return {key: (0, 10) for key in ['A', 'B', 'g', 'k']}

    def get_obs(self) -> dict:
        return self.get_model()(self.get_gt_par())

    def get_gt_par(self) -> dict:
        return {'A': 3, 'B': 1, 'g': 1.5, 'k': 0.5}

    def get_sumstat(self) -> pyabc.Sumstat:
        return pyabc.IdentitySumstat(
            trafos=[
                lambda x: x, lambda x: x**2, lambda x: x**3, lambda x: x**4
            ]
        )