"""
Perturbation
============

Perturbation strategies.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import scipy.stats as st
import copy
from typing import Union

# TODO decide what to do if no parameters there, i.e. if len(X.columns) == 0
# Possible options include
# 1. Have a specific perturber for no particles
# 2. Explicitly hande that case in each concrete perturber implementation
# 3. Make a metaclass which takes care of the no parameters case. This metaclass could also check if w is sane

class Transition(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, w: np.ndarray):
        """
        Fit the perturber to the sampled data.
        Concrete implementations might do something like fitting a KDE.

        Parameters
        ----------
        X: pd.DataFrame
            The parameters.
        w: array
            The corresponding weights
        """

    @abstractmethod
    def rvs(self) -> pd.Series:
        """
        Random variable sample (rvs).

        Sample from the fitted distribution.

        Returns
        -------
        sample: pd.Series
            A sample from the fitted model
        """

    @abstractmethod
    def pdf(self, x: Union[pd.Series, pd.DataFrame]) -> float:
        """
        Evaluate the probability density function (PDF) at x.

        Parameters
        ----------
        x: pd.Series
            Parameter

        Returns
        -------

        density: float
            Probability density at .
        """


class MultivariateNormalTransition(Transition):
    """
    Pretty stupid but should in principle be functional
    """

    def fit(self, X: pd.DataFrame, w: np.ndarray):
        """
        Fit the transition kernel

        Parameters
        ----------
        X: DataFrame
            The parameters.
        w: array
            Array of weights

        It holds that len(X) == len(w).
        """
        if len(X.columns) == 0:
            self.no_parameters = True
        else:
            self.no_parameters = False

        self.X = X
        self.X_arr = X.as_matrix()
        self.w = w  # TODO assert that w is normalized? use metaclass?
        assert np.isclose(w.sum(), 1)
        cov = np.cov(self.X_arr, aweights=w, rowvar=False)
        if len(cov.shape) == 0:
            cov = cov.reshape((1,1))
        self.cov = cov
        import scipy.stats as st
        if not self.no_parameters:
            self.normal = st.multivariate_normal(cov=self.cov)

    def cv(self):
        if not self.no_parameters:
            return variance(self.__class__(), self.X, self.w)

    def rvs(self):
        if self.no_parameters:  # TODO better no parameter handling. metaclass?
            return pd.Series()
        sample = self.X.sample(weights=self.w).iloc[0]
        perturbed = sample + np.random.multivariate_normal(np.zeros(self.cov.shape[0]), self.cov)
        return perturbed

    def pdf(self, x: Union[pd.Series, pd.DataFrame]):
        x = x[self.X.columns]
        x = np.array(x)
        if len(x.shape) == 1:
            x = x[None,:]
        if self.no_parameters:  # TODO better no parameter handling. metaclass?
            return 1
        dens = np.array([(self.normal.pdf(xs - self.X_arr) * self.w).sum() for xs in x])
        return dens if dens.size != 1 else float(dens)


def timeit(f):
    def g(*args):
        import time
        start = time.time()
        res = f(*args)
        end = time.time()
        print(end - start, "s for ", f.__name__, " = ", res)
        return res
    return g


@timeit
def variance(transition: Transition, X: pd.DataFrame, w: np.ndarray):
    """
    Calculate mean coefficient of variation of the KDE.
    """
    transition_cp = copy.copy(transition)
    transition_cp.varprinted = True
    nr_cross_val = 6
    n_samples = len(X)
    uniform_weights = np.ones(n_samples) / n_samples

    density_values = []
    for k in range(nr_cross_val):
        bootstrapped_points = X.sample(n_samples, replace=True, weights=w)
        transition_cp.fit(bootstrapped_points, uniform_weights)
        density_values.append(transition_cp.pdf(X))

    density_values = np.array(density_values)
    variation = st.variation(density_values, 0)
    mean_variation = (variation * w).sum()
    return float(mean_variation)
