from abc import abstractmethod
from typing import Union

import numpy as np
import pandas as pd

from .transitionmeta import TransitionMeta


class Transition(metaclass=TransitionMeta):
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
    @abstractmethod
    def mean_coefficient_of_variation(self) -> float:
        pass

    @abstractmethod
    def required_nr_samples(self, coefficient_of_variation: float) -> int:
        pass