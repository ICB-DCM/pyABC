from typing import Union

import numpy as np
import pandas as pd

from .exceptions import NotEnoughParticles
from .transition import Transition


def scott_rule_of_thumb(n_samples, dimension):
    return n_samples ** (-1. / (dimension + 4))


def silverman_rule_of_thumb(n_samples, dimension):
    return (n_samples * (dimension + 2) / 4.) ** (-1. / (dimension + 4))


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
