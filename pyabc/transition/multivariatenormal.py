from typing import Callable, Union

import numpy as np
import pandas as pd
import scipy.stats as st

from ..parameters import Parameter
from .base import Transition
from .exceptions import NotEnoughParticles
from .util import smart_cov

BandwidthSelector = Callable[[int, int], float]


def scott_rule_of_thumb(n_samples, dimension):
    """
    Scott's rule of thumb.

    .. math::

       \\left ( \\frac{1}{n} \\right ) ^{\\frac{1}{d+4}}

    (see also scipy.stats.kde.gaussian_kde.scotts_factor)
    """
    return n_samples ** (-1.0 / (dimension + 4))


def silverman_rule_of_thumb(n_samples, dimension):
    """
    Silverman's rule of thumb.

    .. math::

       \\left ( \\frac{4}{n (d+2)} \\right ) ^ {\\frac{1}{d + 4}}

    (see also scipy.stats.kde.gaussian_kde.silverman_factor)
    """
    return (4 / n_samples / (dimension + 2)) ** (1 / (dimension + 4))


class MultivariateNormalTransition(Transition):
    """
    Transition via a multivariate Gaussian KDE estimate.

    Parameters
    ----------

    scaling: float
        Scaling is a factor which additionally multiplies the
        covariance with. Since Silverman and Scott usually have too large
        bandwidths, it should make most sense to have 0 < scaling <= 1

    bandwidth_selector: optional
        Defaults to `silverman_rule_of_thumb`.
        The bandwidth selector is a function of the form
        f(n_samples: float, dimension: int),
        where n_samples denotes the (effective) samples size (and is therefore)
        a float and dimension is the parameter dimension.

    """

    def __init__(
        self,
        scaling: float = 1,
        bandwidth_selector: BandwidthSelector = silverman_rule_of_thumb,
    ):
        self.scaling: float = scaling
        self.bandwidth_selector: BandwidthSelector = bandwidth_selector
        # base population as an array
        self._X_arr: Union[np.ndarray, None] = None
        # perturbation covariance matrix
        self.cov: Union[np.ndarray, None] = None
        # normal perturbation distribution
        self.normal = None
        # cache a range array
        self._range = None

    def fit(self, X: pd.DataFrame, w: np.ndarray) -> None:
        if len(X) == 0:
            raise NotEnoughParticles("Fitting not possible.")
        self._X_arr = X.values

        sample_cov = smart_cov(self._X_arr, w)
        dim = sample_cov.shape[0]
        eff_sample_size = 1 / (w**2).sum()
        bw_factor = self.bandwidth_selector(eff_sample_size, dim)

        self.cov = sample_cov * bw_factor**2 * self.scaling
        self.normal = st.multivariate_normal(cov=self.cov, allow_singular=True)

        # cache range array
        self._range = np.arange(len(self.X))

    def rvs(self, size: int = None) -> Union[Parameter, pd.DataFrame]:
        if size is None:
            return self.rvs_single()
        sample_ind = np.random.choice(
            self._range, size=size, p=self.w, replace=True
        )
        sample = self.X.iloc[sample_ind]
        perturbed = sample + np.random.multivariate_normal(
            np.zeros(self.cov.shape[0]), self.cov, size=size
        )
        return perturbed

    def rvs_single(self) -> Parameter:
        sample_ind = np.random.choice(self._range, p=self.w, replace=True)
        sample = self.X.iloc[sample_ind]
        perturbed = sample + np.random.multivariate_normal(
            np.zeros(self.cov.shape[0]), self.cov
        )
        return Parameter(perturbed)

    def pdf(
        self,
        x: Union[Parameter, pd.Series, pd.DataFrame],
    ) -> Union[float, np.ndarray]:
        # convert to numpy array in correct order
        if isinstance(x, (Parameter, pd.Series)):
            x = np.array([x[key] for key in self.X.columns])
        else:
            x = x[self.X.columns].to_numpy()

        # compute density
        if len(x.shape) == 1:
            return self._pdf_single(x)
        else:
            return np.array([self._pdf_single(xi) for xi in x])

        # alternative (higher memory consumption but broadcast)
        # x = np.atleast_3d(x)  # n_sample x n_par x 1
        # dens = self.normal.pdf(np.swapaxes(x - self._X_arr.T, 1, 2)) * self.w
        # dens = np.atleast_2d(dens).sum(axis=1).squeeze()

    def _pdf_single(self, x: np.ndarray):
        return float((self.normal.pdf(x - self._X_arr) * self.w).sum())
