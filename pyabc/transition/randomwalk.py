import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Union

from .base import Transition


class DiscreteRandomWalkTransition(Transition):

    def __init__(self, n_steps: int = 1):
        """
        Parameters
        ----------

        n_steps: int, optional (default = 1)
            Number of random walk steps to take.
        """
        self.n_steps = n_steps

    def fit(self, X: pd.DataFrame, w: np.ndarray):
        pass

    def rvs_single(self) -> pd.Series:
        # take a step
        dim = len(self.X.columns)
        step = do_random_walk(dim, self.n_steps)

        # select a start point
        start_point = self.X.sample(weights=self.w).iloc[0]

        # create randomized point
        perturbed_point = start_point + step
        #print(perturbed_point)
        return perturbed_point

    def pdf(self, x: Union[pd.Series, pd.DataFrame]) \
            -> Union[float, np.ndarray]:
        """
        Evaluate the probability mass function (PMF) at `x`.
        """
        p = 0.0
        for start, weight in zip(self.X.values, self.w):
            p_start = calculate_single_random_walk_probability(start, x, self.n_steps)
            p += p_start * weight
        #print(p)
        return p


def do_random_walk(dim, n_steps):
    state = np.zeros(dim)
    for _ in range(n_steps):
        step = np.random.choice(a=[-1, 1], size=dim)
        state += step
    return state


def calculate_single_random_walk_probability(start, end, n_steps):
    step = end - start
    p = 1.0
    for step_j in step:
        if (step_j + n_steps) % 2 != 0:
            # impossible to get there
            return 0.0
        n_r = int(0.5 * (n_steps + step_j))
        p_j = stats.binom.pmf(n=n_steps, p=0.5, k=n_r)
        p *= p_j
    return p
