import copy
from abc import abstractmethod
from typing import Union
import logging
import numpy as np
import pandas as pd
from scipy import stats as st
from sklearn.base import BaseEstimator
from .exceptions import NotEnoughParticles
from .powerlaw import fitpowerlaw

from .transitionmeta import TransitionMeta

transition_logger = logging.getLogger("Transitions")


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
    NR_STEPS = 10
    FIRST_STEP_FACTOR = 3
    NR_BOOTSTRAP = 5
    X = None
    w = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, w: np.ndarray):
        """
        Fit the perturber to the sampled data.
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
    def rvs(self) -> pd.Series:
        """
        Random variable sample (rvs).

        Sample from the fitted distribution.

        Returns
        -------
        sample: pd.Series
            A sample from the fitted model.
        """

    @abstractmethod
    def pdf(self, x: Union[pd.Series, pd.DataFrame]) \
            -> Union[float, np.ndarray]:
        """
        Evaluate the probability density function (PDF) at `x`.

        Parameters
        ----------
        x: pd.Series
            Parameter

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

    def required_nr_samples(self, coefficient_of_variation: float) -> int:
        if self.no_meaningful_particles():
            raise NotEnoughParticles

        if len(self.X) == 1:
            return 1

        start = max(len(self.X) // self.FIRST_STEP_FACTOR, 1)
        stop = len(self.X) * 2
        step = max(len(self.X) // self.NR_STEPS, 1)

        n_samples_list = list(range(start, stop, step))
        cvs = list(map(self.mean_coefficient_of_variation, n_samples_list))

        self.n_samples_list_ = n_samples_list
        self.cvs_ = cvs

        try:
            popt, f, finv = fitpowerlaw(n_samples_list, cvs)
            self.f_ = f
            self.popt_ = popt
            required_n = finv(coefficient_of_variation)
            return required_n
        except RuntimeError:
            transition_logger.warning("Power law fit failed. "
                                      "Falling back to current nr particles {}"
                                      .format(len(self.X)))
            return len(self.X)

    def mean_coefficient_of_variation(self, n_samples: Union[None, int]=None) \
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
        if self.no_meaningful_particles():
            raise NotEnoughParticles(n_samples)

        if n_samples is None:
            n_samples = len(self.X)

        self_cp = copy.copy(self)
        uniform_weights = np.ones(n_samples) / n_samples

        # TODO: decide which test points to use
        # maybe also sample them from the kde directly?
        # however X and w might better represent the next population.
        # not sure what is best
        test_points = self.X
        test_weights = self.w

        bootstrapped_pdfs_at_test = []
        for k in range(self.NR_BOOTSTRAP):
            bootstrapped_points = pd.DataFrame(
                [self.rvs() for _ in range(len(uniform_weights))])
            self_cp.fit(bootstrapped_points, uniform_weights)
            bootstrapped_pdfs_at_test.append(self_cp.pdf(test_points))

        bootstrapped_pdfs_at_test = np.array(bootstrapped_pdfs_at_test)
        variation_at_test = st.variation(bootstrapped_pdfs_at_test, 0)
        mean_variation = (variation_at_test * test_weights).sum()
        if not np.isfinite(mean_variation):
            msg = "CV not finite {}".format(mean_variation)
            raise NotEnoughParticles(msg)

        self.test_points_ = test_points
        self.test_weights_ = test_weights
        self.variation_at_test_points_ = variation_at_test
        return mean_variation
