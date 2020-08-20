"""A discrete jump transition function."""

import numpy as np
import pandas as pd
from typing import Union

from ..random_choice import fast_random_choice
from ..random_variables import RV
from .base import DiscreteTransition


class PerturbationKernel:
    """Parameter perturbation kernel for a discrete set of parameters.

    Parameters
    ----------
    domain:
        All possible parameter values.
    p_stay:
        The probability to stay at a given parameter value.
    """

    def __init__(self, domain: np.ndarray, p_stay: float = 0.7):
        if len(np.unique(domain)) != len(domain):
            raise ValueError("The domain contains duplicates.")
        self.domain = domain

        if len(domain) == 1:
            p_stay = 1.
        if not 0 <= p_stay <= 1:
            raise ValueError("p_stay must be in [0, 1].")
        self.p_stay = p_stay
        self.p_move = (1 - p_stay) / (len(self.domain) - 1)

        # cache a random variable (later the start index and 0 must be swapped)
        indices = np.arange(len(domain))
        probabilities = [p_stay, *[self.p_move] * (len(self.domain) - 1)]
        self.rv = RV('rv_discrete', values=(indices, probabilities))

    def rvs(self, a: float) -> float:
        """Sample a kernel jump from parameter `a` to another parameter."""
        if a not in self.domain:
            raise ValueError("The parameter value is not in the domain.")
        if len(self.domain) == 1:
            return a

        # sample from the cached random variable
        ix = self.rv.rvs()

        # get index of the starting parameter
        ix_a = np.argmax(self.domain == a)

        # in the cached rv, ix_a and 0 were swapped -> swap them again
        if ix == 0:
            ix = ix_a
        elif ix == ix_a:
            ix = 0

        return self.domain[ix]

    def pdf(self, b: float, a: float) -> float:
        """Probability mass function for a jump to target `b` from source `a`.
        """
        if a not in self.domain or b not in self.domain:
            raise ValueError(
                "At least one parameter value is not in the domain.")
        return self.p_stay if b == a else self.p_move


class DiscreteJumpTransition(DiscreteTransition):
    """
    Transition with positive random jump probability for discrete parameters.
    Adapts base draw probabilities to the last generation's histogram and then
    jumps to an arbitrary other parameter with a positive jump probability to
    ensure that the prior is absolutely continuous w.r.t. the proposal.

    Parameters
    ----------
    domain, p_stay:
        See the PerturbationKernel.

    .. note::
        This transition can only deal with a single parameter. Use an
        AggregatedTransition to combine multiple parameters.
    """

    def __init__(self, domain: np.ndarray,
                 p_stay: float = 0.7):
        self.values = None
        self.weights = None
        self.perturbation_kernel = PerturbationKernel(
            domain=domain, p_stay=p_stay)

    def fit(self, X: pd.DataFrame, w: np.ndarray) -> None:
        """Fit starting weights to the distribution of samples."""
        # this is only meant to be used with a single parameter
        if len(X.columns) != 1:
            raise ValueError(
                "This transition can only handle a single parameter.")
        # compute a single weight per unique parameter value
        x = np.array(X).flatten()
        self.values = []
        self.weights = []
        for value in np.unique(x):
            self.values.append(value)
            self.weights.append(sum(w[x == value]))
        self.weights = np.array(self.weights)
        self.weights /= self.weights.sum()

    def rvs_single(self) -> pd.Series:
        """Generate a single random variable."""
        # sample a starting index
        index = fast_random_choice(self.weights)
        # get value at that index
        value = self.values[index]
        # maybe jump to another value
        value = self.perturbation_kernel.rvs(value)
        return pd.Series({self.X.columns[0]: value})

    def pdf(self, x: Union[pd.Series, pd.DataFrame]) \
            -> Union[float, np.ndarray]:
        """Compute the probability mass function at `x`."""
        x = float(np.array(x))
        return sum(w * self.perturbation_kernel.pdf(x, start)
                   for w, start in zip(self.weights, self.values))
