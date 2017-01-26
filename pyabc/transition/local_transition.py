import numpy.linalg as la
import scipy as sp
import pandas as pd
from .base import Transition
from scipy.spatial import cKDTree
from .util import smart_cov
from .exceptions import NotEnoughParticles


class LocalTransition(Transition):
    """
    Local KDE fit.

    Parameters
    ----------

    k: int
        Number of nearest neighbors for local covariance
        calculation.

    scaling: float
        Scaling factor for the local covariance matrices.
    """

    def __init__(self, k=50, scaling=1):
        self.k = k
        self.scaling = scaling

    def fit(self, X, w):
        if len(X) == 0:
            raise NotEnoughParticles("Fitting not possible.")
        self.X_arr = X.as_matrix()

        ctree = cKDTree(X)
        _, indices = ctree.query(X, k=min(self.k + 1, X.shape[0]))

        self.covs = sp.array([self._calc_cov(n, indices)
                              for n in range(X.shape[0])])
        self.inv_covs = sp.array(list(map(la.inv, self.covs)))
        self.determinants = sp.array(list(map(la.det, self.covs)))
        self.normalization = sp.sqrt(
            (2 * sp.pi) ** self.X_arr.shape[1] * self.determinants)

    def pdf(self, x):
        x = x[self.X.columns].as_matrix()
        if len(x.shape) == 1:
            return self._pdf_single(x)
        else:
            return sp.array([self._pdf_single(x) for x in x])

    def _pdf_single(self, x):
        distance = self.X_arr - x
        cov_distance = sp.einsum("ijk,ik,ij->i", self.inv_covs, distance,
                                 distance)
        return (sp.exp(-.5 * cov_distance) / self.normalization * self.w).sum()

    def _calc_cov(self, n, indices):
        """
        Calculate covariance around local support vector
        """
        if len(indices) > 1:
            surrounding_indices = indices[n, 1:]
            nearest_vector_deltas = (self.X_arr[surrounding_indices]
                                     - self.X_arr[n])
            local_weights = self.w[surrounding_indices]
        else:
            nearest_vector_deltas = sp.absolute(self.X_arr)
            local_weights = sp.array([1])

        cov = smart_cov(nearest_vector_deltas,
                        local_weights / local_weights.sum())
        if sp.absolute(cov.sum()) == 0:
            for k in range(cov.shape[0]):
                cov[k, k] = sp.absolute(self.X_arr[0, k])
        return cov * self.scaling

    def rvs(self):
        support_index = sp.random.choice(self.w.shape[0], p=self.w)
        sample = sp.random.multivariate_normal(self.X_arr[support_index],
                                               self.covs[support_index])
        return pd.Series(sample, index=self.X.columns)
