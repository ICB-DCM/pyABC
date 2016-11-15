import copy
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats as st

from .powerlaw import fitpowerlaw
from .exceptions import NotEnoughParticles
from .transition import Transition


class MultivariateNormalTransition(Transition):
    """
    Pretty stupid but should in principle be functional
    """
    NR_CROSS_VAL = 10
    START_FACTOR = 3
    NR_STEPS = 30

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
        if len(X) == 0:
            raise NotEnoughParticles("Fitting not possible.")

        self.X = X
        self.X_arr = X.as_matrix()
        self.w = w
        if len(X) > 1:
            cov = np.cov(self.X_arr, aweights=w, rowvar=False)
        else:
            cov_diag = self.X_arr[0]
            cov = np.diag(cov_diag)
        if len(cov.shape) == 0:
            cov = cov.reshape((1,1))
        self.cov = cov * scott_rule_of_thumb(len(X) / (1 + w.var()), cov.shape[0])
        import scipy.stats as st
        if not self.no_parameters:
            self.normal = st.multivariate_normal(cov=self.cov, allow_singular=True)

    def required_nr_samples(self, coefficient_of_variation):
        if not hasattr(self, "X") or not hasattr(self, "w"):
            raise NotEnoughParticles

        if len(self.X) == 1:
            return lambda x: 1

        start = max(len(self.X) // self.START_FACTOR, 1)
        stop = len(self.X)
        step = max(len(self.X) // self.NR_STEPS, 1)

        n_samples_list = list(range(start, stop, step)) + [len(self.X)]
        cvs = list(map(self.mean_coefficient_of_variation, n_samples_list))

        popt, f, finv = fitpowerlaw(n_samples_list, cvs)
        required_n = finv(coefficient_of_variation)
        return required_n

    def rvs(self):
        sample = self.X.sample(weights=self.w).iloc[0]
        perturbed = sample + np.random.multivariate_normal(np.zeros(self.cov.shape[0]), self.cov)
        return perturbed

    def pdf(self, x: Union[pd.Series, pd.DataFrame]):
        x = x[self.X.columns]
        x = np.array(x)
        if len(x.shape) == 1:
            x = x[None,:]
        dens = np.array([(self.normal.pdf(xs - self.X_arr) * self.w).sum() for xs in x])
        return dens if dens.size != 1 else float(dens)

    def mean_coefficient_of_variation(self, n_samples=None):
        if not self.no_parameters:
            if not hasattr(self, "X") or not hasattr(self, "w"):
                raise NotEnoughParticles

        if n_samples is None:
            n_samples = len(self.X)

        self_cp = copy.copy(self)
        uniform_weights = np.ones(n_samples) / n_samples

        density_values = []
        for k in range(self_cp.NR_CROSS_VAL):
            bootstrapped_points = self.X.sample(n_samples, replace=True, weights=self.w)
            self_cp.fit(bootstrapped_points, uniform_weights)
            density_values.append(self_cp.pdf(self.X))

        density_values = np.array(density_values)
        variation = st.variation(density_values, 0)
        mean_variation = (variation * self.w).sum()
        return mean_variation


def scott_rule_of_thumb(n_samples, dimension):
    return n_samples ** (-1. / (dimension + 4))


def silverman_rule_of_thumb(n_samples, dimension):
    return (n_samples * (dimension + 2) / 4.) ** (-1. / (dimension + 4))


