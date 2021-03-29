"""Models from Prangle 2015, Adapting the ABC Distance Function."""

import numpy as np
from typing import Callable

from pyabc import Distribution, RV
from .base import Problem, gk


class PrangleNormalProblem(Problem):

    def get_model(self) -> Callable:
        def model(p):
            """One informative, one uninformative statistic"""
            return {
                's1': p['theta'] + 0.1 * np.random.normal(),
                's2': 1 * np.random.normal(),
            }
        return model

    def get_prior(self) -> Distribution:
        prior = Distribution(theta=RV('norm', loc=0, scale=100))
        return prior

    def get_prior_bounds(self) -> dict:
        return {'theta': (0, 100)}

    def get_obs(self) -> dict:
        return {'s1': 0, 's2': 0}

    def get_gt_par(self) -> dict:
        return {'theta': 0}


class PrangleGKProblem(Problem):

    def get_model(self) -> Callable:
        def model(p):
            A, B, g, k = [p[key] for key in ['A', 'B', 'g', 'k']]
            c = 0.8
            vals = gk(A=A, B=B, c=c, g=g, k=k, n=10000)
            ordered = np.sort(vals)
            subset = ordered[1250:8751:1250]
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


class PrangleLVProblem(Problem):

    def get_model(self) -> Callable:
        import ssa

        reactants = np.array([[1, 0],
                              [1, 1],
                              [0, 1]])
        products = np.array([[2, 0],
                             [0, 2],
                             [0, 0]])

        x0 = np.array([50, 100])

        t_max = 32
        # output = ssa.output.FullOutput()
        ts = np.arange(2, 32, 2)
        output = ssa.output.ArrayOutput(ts=ts)
        sigma = np.exp(2.3)

        def model(p):
            k = np.array([p['p1'], p['p2'], p['p3']])
            # log parameters
            k = np.exp(k)
            ssa_model = ssa.Model(
                reactants, products, x0=x0,
                k=k, t_max=t_max, max_reactions=1e5, output=output)
            try:
                ret = ssa_model.simulate()
                # shape: (t, 2)
                sims = ret.list_xs[0]
            except ValueError:
                sims = np.empty((ts.size, 2))
                sims[:] = np.nan

            return {
                'y': sims + sigma * np.random.normal(size=sims.shape),
            }
        return model

    def get_prior(self) -> Distribution:
        return Distribution(
            p1=RV('uniform', -6, 8),
            p2=RV('uniform', -6, 8),
            p3=RV('uniform', -6, 8))

    def get_prior_bounds(self) -> dict:
        return {key: (-6, 2) for key in ['p1', 'p2', 'p3']}

    def get_obs(self) -> dict:
        return self.get_model()(self.get_gt_par())

    def get_gt_par(self) -> dict:
        return {'p1': np.log(1), 'p2': np.log(0.005), 'p3': np.log(0.6)}
