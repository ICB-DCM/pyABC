import numpy.linalg as la
import scipy as sp
from .base import Transition
from scipy.spatial import cKDTree


class LocalTransition(Transition):
    """

    Parameters
    ----------

    k: int
        Number of nearest neighbors for local covariance
        calculation.

    scaling: float
        Scaling factor for the local covariance matrices.
    """

    def __init__(self, k=10, scaling=1):
        self.k = k
        self.scaling = scaling

    def fit(self, X, w):
        self.X = X
        self.w = w

        ctree = cKDTree(X)
        _, indices = ctree.query(X, k=min(self.k + 1, X.shape[0]))

        self.covs = sp.array([self._calc_cov(n, indices)
                              for n in range(X.shape[0])])
        self.inv_covs = sp.array(list(map(la.inv, self.covs)))
        self.determinants = sp.array(list(map(la.det, self.covs)))
        self.normalization = sp.sqrt(
            (2 * sp.pi) ** self.X.shape[1] * self.determinants)

    def pdf(self, x):
        distance = self.X - x
        cov_distance = sp.einsum("ijk,ik,ij->i", self.inv_covs, distance,
                                 distance)
        return (sp.exp(-.5 * cov_distance) / self.normalization * self.w).sum()

    def _calc_cov(self, n, indices):
        """
        Calculate covariance of around local support vector
        """
        indices_excluding_self = indices[n, 1:]
        nearest_vectors = self.X[indices_excluding_self] - self.X[n]
        local_weights = self.w[indices_excluding_self]
        cov = sp.cov(nearest_vectors, rowvar=False,
                     aweights=local_weights / local_weights.sum())
        return cov * self.scaling

    def rvs(self):
        support_index = sp.random.choice(self.w.shape[0], p=self.w)
        sample = sp.random.multivariate_normal(self.X[support_index],
                                               self.covs[support_index])
        return sample
