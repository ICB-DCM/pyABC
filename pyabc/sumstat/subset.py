"""Identification of sample subsets for model training."""

import numpy as np
from typing import Tuple
import logging
from abc import ABC, abstractmethod

from ..weighted_statistics import weighted_mean

try:
    import sklearn.mixture as skl_mx
except ImportError:
    skl_mx = None


logger = logging.getLogger("ABC.Sumstat")


class Subsetter(ABC):
    """Select a localized sample subset for model training.

    E.g. in the :class:`pyabc.PredictorSumstat` class, we employ predictor
    models `y -> p` from data to parameters.
    These models should be local, e.g. trained on samples from a high-density
    region. This is because the inverse mapping of `p -> y`, `y -> p`, does
    in general not exist globally, e.g. due to parameter non-identifiability,
    multiple modes, and model stochasticity. Therefore, it is important to
    train the models on a sample set in which a functional form is roughly
    given.
    This class allows to subset a given sample to generate a localized sample.
    """

    @abstractmethod
    def select(
            self,
            x: np.ndarray,
            y: np.ndarray,
            w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select samples for model training. This is the main method.

        Parameters
        ----------
        x: Samples, shape (n_sample, n_feature).
        y: Targets, shape (n_sample, n_out).
        w: Weights, shape (n_sample,).

        Returns
        -------
        x_, y_, w_:
            Subsetted samples, targets and weights with
            n_sample -> n_sample_used <= n_sample.
        """


class IdSubsetter(Subsetter):
    """Identity subset mapping."""

    def select(
            self,
            x: np.ndarray,
            y: np.ndarray,
            w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Just return x, y, w unchanged."""
        return x, y, w


class GMMSubsetter:
    """Using a Gaussian mixed model for subset identification.

    Performs model selection over Gaussian mixed models with up to
    `n_components_max` components and returns all samples belonging to the same
    cluster as the posterior mean.
    Optionally, this set is augmented by the nearest neighbors to reach a
    fraction `min_fraction` of the original sample size.

    Parameters
    ----------
    n_components_min: Minimum candidate number of clusters.
    n_components_max: Maximum candidate number of clusters.
    min_fraction:
        Minimum fraction of samples in the result. If the obtained cluster has
        less samples, it is augmented by nearby samples.
    gmm_args:
        Keyword arguments that are passed on to the sklearn `GaussianMixture`,
        see
        https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html.  # noqa

    Properties
    ----------
    gmm: The best fitted Gaussian mixture model.
    n_components: The corresponding number of components.
    bics: All BIC values used in model selection.
    """

    def __init__(
        self,
        n_components_min: int = 1,
        n_components_max: int = 5,
        min_fraction: float = 0.3,
        gmm_args: dict = None,
    ):
        if skl_mx is None:
            raise ImportError(
                "This class requires an installation of scikit-learn. "
                "Install e.g. via `pip install pyabc[scikit-learn]`")

        self.n_components_min: int = n_components_min
        self.n_components_max: int = n_components_max
        self.min_fraction: float = min_fraction

        if gmm_args is None:
            gmm_args = {}
        default_gmm_args = {'max_iter': 500, 'n_init': 5}
        default_gmm_args.update(gmm_args)
        self.gmm_args: dict = default_gmm_args

        # status variables, filled in `select()`
        self.gmm = None
        self.n_components = None
        self.bics = None

    def select(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select based on GMM clusters.

        Parameters
        ----------
        x: Samples, shape (n_sample, n_feature).
        y: Targets, shape (n_sample, n_out).
        w: Weights, shape (n_sample,).
        """
        # find most suitable number of clusters
        gmm, n_components, bics = self._find_optimal_model(y=y)
        self.gmm = gmm
        self.n_components = n_components
        self.bics = bics

        # identify cluster of posterior mean
        mean: np.ndarray = weighted_mean(y, w / w.sum()).reshape(1, -1)
        cluster_id: np.ndarray = gmm.predict(mean)

        # find cluster associations of data points
        cluster_ids: np.ndarray = gmm.predict(y)
        in_cluster: np.ndarray = cluster_ids == cluster_id

        x_new, y_new, w_new = get_augmented_subset(
            x=x, y=y, w=w, ref=mean, in_cluster=in_cluster,
            min_fraction=self.min_fraction,
        )

        logger.info(
            f"Subsetting: #clusters: {n_components}, "
            f"target cluster points: {sum(in_cluster)}, using {len(y_new)} "
            f"(BICs: {bics})",
        )

        return x_new, y_new, w_new

    def _find_optimal_model(self, y: np.ndarray):
        """Find optimal GMM model.

        Parameters
        ----------
        y: The data to fit the model on (usually parameters).

        Returns
        -------
        best_gmm, best_n_components, bics:
            The best fitted GMM, the number of components, and all BIC values.
        """

        # iterate over numbers of clusters
        bics = {}
        best_gmm = None
        best_n_components = None
        for n_components in range(
                self.n_components_min, self.n_components_max+1):
            # fit a Gaussian mixture model
            gmm = skl_mx.GaussianMixture(
                n_components=n_components, **self.gmm_args)
            gmm.fit(y)

            # evaluate BIC as model selection criterion
            bic = gmm.bic(y)
            if len(bics) == 0 or bic < min(bics.values()):
                best_gmm = gmm
                best_n_components = n_components
            bics[n_components] = bic

        return best_gmm, best_n_components, bics


def get_augmented_subset(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, ref: np.ndarray,
    in_cluster: np.ndarray, min_fraction: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select indices from clustering and potential augment to match minimum.

    Parameters
    ----------
    x, y, w: The original x, y, w.
    ref: Reference value (e.g. posterior mean).
    in_cluster: Boolean indicators whether an entry is in the cluster.
    min_fraction: Minimum required fraction of particles, in [0, 1].

    Returns
    -------
    x_new, y_new, w_new: The subset entries.
    """
    x_new, y_new, w_new = x[in_cluster], y[in_cluster], w[in_cluster]

    desired: int = int(min_fraction * len(y))
    ixs_in_cluster: np.ndarray = np.flatnonzero(in_cluster)
    required: int = desired - len(ixs_in_cluster)

    if required <= 0:
        return x_new, y_new, w_new

    # sort remaining values by distance to reference point
    x_left, y_left, w_left = x[~in_cluster], y[~in_cluster], w[~in_cluster]
    distances: np.ndarray = np.linalg.norm(y_left - ref, ord=2, axis=1)
    # indices of the required closest parameters
    ixs_nearest: np.ndarray = np.argpartition(distances, required)[:required]

    # select both clustering and smallest distance entries
    x_new = np.row_stack((x_new, x_left[ixs_nearest]))
    y_new = np.row_stack((y_new, y_left[ixs_nearest]))
    w_new = np.row_stack((w_new, w_left[ixs_nearest]))

    if any(len(val) != desired for val in [x_new, y_new, w_new]):
        raise AssertionError("Unexpected number of entries.")

    return x_new, y_new, w_new
