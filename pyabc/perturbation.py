"""
Perturbation
============

Perturbation strategies.
"""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class ParticlePerturber(ABC):
    @abstractmethod
    def fit(self, X: pd.DataFrame, w: np.ndarray):
        pass

    @abstractmethod
    def rvs(self):
        pass


class IndependentNormalPerturber(ParticlePerturber):
    """
    Pretty stupid but should in principle be more ore less functional
    """

    def fit(self, X: pd.DataFrame, w: np.ndarray):
        """
        Fit the perturberer

        Parameters
        ----------
        X: DataFrame
            The parameters.
        w: array
            Array of weights

        It holds that len(X) == len(w).
        """
        self.X = X
        self.w = w
        self.std = X.std()

    def rvs(self) -> pd.Series:
        sample = self.X.sample(weights=self.w).iloc[0]
        perturbed = sample + self.std * np.random.randn(len(self.std))
        return perturbed
