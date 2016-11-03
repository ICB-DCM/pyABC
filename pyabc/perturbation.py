"""
Perturbation
============

Perturbation strategies.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


# TODO decide what to do if no parameters there, i.e. if len(X.columns) == 0
# Possible options include
# 1. Have a specific perturber for no particles
# 2. Explicitly hande that case in each concrete perturber implementation
# 3. Make a metaclass which takes care of the no parameters case

class ParticlePerturber(ABC):
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
    def pdf(self, x: pd.Series) -> float:
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


class MultivariateNormalPerturber(ParticlePerturber):
    """
    Pretty stupid but should in principle be more ore less functional
    """

    def fit(self, X: pd.DataFrame, w: np.ndarray):
        """
        Fit the perturberer

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
        self.w = w

        cov = np.cov(self.X_arr, aweights=w, rowvar=False)
        if len(cov.shape) == 0:
            cov = cov.reshape((1,1))
        self.cov = cov

    def rvs(self):
        if self.no_parameters:
            return pd.Series()
        sample = self.X.sample(weights=self.w).iloc[0]
        perturbed = sample + np.random.multivariate_normal(np.zeros(self.cov.shape[0]), self.cov)
        return perturbed

    def pdf(self, x: pd.Series):
        if self.no_parameters:
            return 1
        import scipy.stats as st
        pdf = st.multivariate_normal(cov=self.cov).pdf
        dens = sum(w * pdf(x) for w, x in zip(self.w, self.X_arr))
        return float(dens)
