from typing import Callable
import numpy as np
import pyabc


class Problem:

    def get_model(self) -> Callable:
        """Get the model."""

    def get_prior(self) -> pyabc.Distribution:
        """Get the prior."""

    def get_prior_bounds(self) -> dict:
        """Get prior boundaries"""

    def get_obs(self) -> dict:
        """Get the observation."""

    def get_gt_par(self) -> dict:
        """Get the ground truth parameters."""

    def get_sumstat(self) -> pyabc.Sumstat:
        """Get summary statistic function."""
        return pyabc.IdentitySumstat()


def gk(A, B, c, g, k, n: int = 1):
    """One informative, one uninformative statistic"""
    z = np.random.normal(size=n)
    e = np.exp(- g * z)
    return A + B * (1 + c * (1 - e) / (1 + e)) * (1 + z**2)**k * z