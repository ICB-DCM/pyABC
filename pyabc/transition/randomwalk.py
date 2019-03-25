import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Union

from .base import DiscreteTransition


class DiscreteRandomWalkTransition(DiscreteTransition):
    """
    This transition is based on a discrete random walk. This may be useful
    for discrete ordinal parameter distributions that can be described as
    lieing on the grid of integers.

    .. note::
        This transition does not adapt to the problem structure and thus has
        potentially slow convergence.
        Further, the transition does not satisfy proposal >> prior, so that
        it is indeed not valid as an importance sampling distribution. This
        can be overcome by selecting the number of steps as a random variable.

    Parameters
    ----------

    n_steps: int, optional (default = 1)
        Number of random walk steps to take.
    """

    def __init__(self,
                 n_steps: int = 1,
                 p_l: float = 1. / 3,
                 p_r: float = 1. / 3,
                 p_c: float = 1. / 3):
        self.n_steps = n_steps
        self.p_l = p_l
        self.p_r = p_r
        self.p_c = p_c

    def fit(self, X: pd.DataFrame, w: np.ndarray):
        pass

    def rvs_single(self) -> pd.Series:
        # take a step
        dim = len(self.X.columns)
        step = perform_random_walk(
            dim, self.n_steps, self.p_l, self.p_r, self.p_c)

        # select a start point
        start_point = self.X.sample(weights=self.w).iloc[0]

        # create randomized point
        perturbed_point = start_point + step

        return perturbed_point

    def pdf(self, x: Union[pd.Series, pd.DataFrame]) \
            -> Union[float, np.ndarray]:
        """
        Evaluate the probability mass function (PMF) at `x`.
        """
        if not np.all(np.isclose(x, x.astype(int))):
            raise ValueError(
                f"Transition can only handle integer values, not fulfilled "
                f"by x={x}.")
        x = x[self.X.columns]
        x = np.array(x)
        if len(x.shape) == 1:
            return self.pdf_single(x)
        else:
            return np.array([self.pdf_single(x_) for x_ in x])

    def pdf_single(self, x):
        p = 0.0
        for start, weight in zip(self.X.values, self.w):
            # probability if started from start
            p_start = calculate_single_random_walk_probability(
                start, x, self.n_steps, self.p_l, self.p_r, self.p_c)
            # add p_start times the weight associated to p_start
            p += p_start * weight

        return p


def perform_random_walk(dim, n_steps, p_l, p_r, p_c):
    """
    Perform a random walk in [-1, 0, 1] in each dimension, for `n_steps`
    steps.
    """
    state = np.zeros(dim)
    for _ in range(n_steps):
        # sample a step
        step = np.random.choice(a=[-1, 0, 1], p=[p_l, p_c, p_r], size=dim)
        state += step
    return state


def calculate_single_random_walk_probability(
        start, end, n_steps,
        p_l: float = 1. / 3, p_r: float = 1. / 3, p_c: float = 1. / 3):
    """
    Calculate the probability of getting from state `start` to state `end`
    in `n_steps` steps, where the probabilities for a left, right, and
    no step are `p_l`, `p_r`, `p_c`, respectively.
    """
    step = end - start
    p = 1.0
    for step_j in step:
        p_j = 0.0
        for n_r in range(max(int(step_j), 0), n_steps + 1):
            n_l = n_r - step_j
            n_c = n_steps - n_r - n_l
            p_j += stats.multinomial.pmf(
                x=[n_l, n_r, n_c], n=n_steps, p=[p_l, p_r, p_c])
        p *= p_j
    return p


def calculate_single_random_walk_probability_no_stay(start, end, n_steps):
    """
    Calculate the probability of getting from state `start` to state `end`
    in `n_steps` steps. Simplified formula assuming the probability to remain
    in a given state is zero in each iteration, i.e. that in every step
    there is a move to the left or right.
    Note that the iteration of this transition is not surjective on the grid
    in dimension dim >= 2.
    """
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
