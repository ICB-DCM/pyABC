from abc import abstractmethod
from typing import Union
import logging
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from .exceptions import NotEnoughParticles
from .predict_population_size import predict_population_size
from ..cv.bootstrap import calc_cv
from .transitionmeta import TransitionMeta

logger = logging.getLogger("Transitions")


class Transition(BaseEstimator, metaclass=TransitionMeta):
    """
    Abstract Transition base class. Derive all Transitions from this class

        .. note::
            This class does a little bit of meta-programming.

            The `fit`, `pdf` and `rvs` methods are automatically wrapped
            to handle the special case of no parameters.

            Hence, you can safely assume that you encounter at least one
            parameter. All the defined transitions will then automatically
            generalize to the case of no paramter.
    """
    NR_BOOTSTRAP = 5
    X = None
    w = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, w: np.ndarray):
        """
        Fit the density estimator (perturber) to the sampled data.
        Concrete implementations might do something like fitting a KDE.

        The parameters given as ``X`` and ``w`` are automatically stored
        in ``self.X`` and ``self.w``.

        Parameters
        ----------
        X: pd.DataFrame
            The parameters.
        w: array
            The corresponding weights
        """

    @abstractmethod
    def rvs_single(self) -> pd.Series:
        """
        Random variable sample (rvs).

        Sample from the fitted distribution.

        Returns
        -------
        sample: pd.Series
            A sample from the fitted model.
        """

    def rvs(self, size=None):
        """
        Sample from the density.

        Parameters
        ----------

        size: int, optional
            Number of independent samples to draw.
            Defaults to 1 and is in this case equivalent to calling
            "rvs_single".

        Returns
        -------

        samples: The samples as pandas DataFrame


        Note
        ----

        This method can be overridden for efficient implementations.
        The default is to call rvs_single repeatedly (which might
        not be the most efficient way).

        """
        if size is None:
            return self.rvs_single()
        return pd.DataFrame([self.rvs_single() for _ in range(size)])

    @abstractmethod
    def pdf(self, x: Union[pd.Series, pd.DataFrame]) \
            -> Union[float, np.ndarray]:
        """
        Evaluate the probability density function (PDF) at `x`.

        Parameters
        ----------
        x: pd.Series, pd.DataFrame
            Parameter. If x is a series, then x should have the the columns
            from X passed to the fit method as indices.
            If x is a DataFrame, then x should have the same columns as X
            passed before to the fit method. The order of the columns is not
            important

        Returns
        -------

        density: float
            Probability density at `x`.
        """

    def score(self, X: pd.DataFrame, w: np.ndarray):
        densities = self.pdf(X)
        return (np.log(densities) * w).sum()

    def no_meaningful_particles(self) -> bool:
        return len(self.X) == 0 or self.no_parameters

    def mean_cv(self, n_samples: Union[None, int] = None) \
            -> float:
        """
        Estimate the uncertainty on the KDE.

        Parameters
        ----------
        n_samples: int, optional
            Estimate the CV for ``n_samples`` samples.
            If this parameter is not given, the sample size of the last fit
            is used.

        Returns
        -------

        mean_cv: float
            The estimated average coefficient of variation.

        Note
        ----

        A call to this method, as a side effect, also sets the attributes
        ``test_points_``, ``test_weights_`` and ``variation_at_test_points_``.
        These are the individual points, weights and varations
        used to calculate the mean.
        """
        # TODO: not sure if this is the right behaviour
        if self.no_meaningful_particles():
            raise NotEnoughParticles(n_samples)

        if n_samples is None:
            n_samples = len(self.X)

        test_points = self.X
        test_weights = self.w

        self.test_points_ = test_points
        self.test_weights_ = test_weights

        cv, variation_at_test = calc_cv(n_samples, np.array([1]),
                                        self.NR_BOOTSTRAP, test_weights,
                                        [self], [test_points])

        self.variation_at_test_points_ = variation_at_test[0]
        return cv

    def required_nr_samples(self, coefficient_of_variation: float) -> int:
        if self.no_meaningful_particles():
            raise NotEnoughParticles

        res = predict_population_size(len(self.X), coefficient_of_variation,
                                      self.mean_cv)
        self.cv_estimate_ = res
        return res.n_estimated


class DiscreteTransition(Transition):
    """
    This is a base class for discrete transition kernels.
    """
