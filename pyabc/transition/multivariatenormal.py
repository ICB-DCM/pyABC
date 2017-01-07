from typing import Union

import numpy as np
import pandas as pd
import scipy.stats as st
from .exceptions import NotEnoughParticles
from .transition import Transition


def scott_rule_of_thumb(n_samples, dimension):
    return n_samples ** (-1. / (dimension + 4))


def silverman_rule_of_thumb(n_samples, dimension):
    return (4 / n_samples / (dimension + 2)) ** (1 / (dimension + 4))


class MultivariateNormalTransition(Transition):
    """
    Pretty stupid but should in principle be functional

    Parameters
    ----------

    scaling: float
        Scaling is a factor which additionally multiplies the
        bandwidth with. Since silverman and Scott usually have to large
        bandwidths, it should make most sense to have 0 < scaling <= 1

    """
    def __init__(self, scaling=1.):
        self.scaling = scaling

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

        self._X_arr = X.as_matrix()
        cov = smart_cov(self._X_arr, w)
        effective_sample_size = len(X) / (1 + w.var())
        dimension = cov.shape[0]
        self.cov = cov * silverman_rule_of_thumb(effective_sample_size, dimension) * self.scaling
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
        dens = np.array([(self.normal.pdf(xs - self._X_arr) * self.w).sum() for xs in x])
        return dens if dens.size != 1 else float(dens)


def smart_cov(X_arr, w):
    """
    Also returns a covariance of X_arr consists of only one single sample
    """
    if X_arr.shape[0] == 1:
        cov_diag = X_arr[0]
        cov = np.diag(np.absolute(cov_diag))
        return cov

    cov = np.cov(X_arr, aweights=w, rowvar=False)
    if len(cov.shape) == 0:
        cov = np.array([[cov]])
    return cov
