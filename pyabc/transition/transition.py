import copy
from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats as st

from .exceptions import NotEnoughParticles
from .powerlaw import fitpowerlaw

from .transitionmeta import TransitionMeta


class Transition(metaclass=TransitionMeta):
    NR_STEPS = 30
    FIRST_STEP_FACTOR = 3
    NR_BOOTSTRAP = 10

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
    def pdf(self, x: Union[pd.Series, pd.DataFrame]) -> Union[float, np.ndarray]:
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

    def required_nr_samples(self, coefficient_of_variation):
        if not hasattr(self, "X") or not hasattr(self, "w"):
            raise NotEnoughParticles

        if len(self.X) == 1:
            return lambda x: 1

        start = max(len(self.X) // self.FIRST_STEP_FACTOR, 1)
        stop = len(self.X)
        step = max(len(self.X) // self.NR_STEPS, 1)

        n_samples_list = list(range(start, stop, step)) + [len(self.X)]
        cvs = list(map(self.mean_coefficient_of_variation, n_samples_list))

        popt, f, finv = fitpowerlaw(n_samples_list, cvs)
        required_n = finv(coefficient_of_variation)
        return required_n

    def mean_coefficient_of_variation(self, n_samples=None):
        if not self.no_parameters:
            if not hasattr(self, "X") or not hasattr(self, "w"):
                raise NotEnoughParticles

        if n_samples is None:
            n_samples = len(self.X)

        self_cp = copy.copy(self)
        uniform_weights = np.ones(n_samples) / n_samples

        density_values = []
        for k in range(self.NR_BOOTSTRAP):
            bootstrapped_points = self.X.sample(n_samples, replace=True, weights=self.w)
            self_cp.fit(bootstrapped_points, uniform_weights)
            density_values.append(self_cp.pdf(self.X))

        density_values = np.array(density_values)
        variation = st.variation(density_values, 0)
        mean_variation = (variation * self.w).sum()
        return mean_variation
