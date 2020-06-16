"""
Weighted statistics
===================

Functions performing statistical operations on weighted points
generated via importance sampling.
"""

import numpy as np
from functools import wraps


def weight_checked(function):
    """
    Function decorator to check normalization of weights.
    """
    @wraps(function)
    def function_with_checking(points, weights=None, **kwargs):
        if weights is not None and not np.isclose(weights.sum(), 1):
            raise AssertionError(
                f"Weights not normalized: {weights.sum()}.")
        return function(points, weights, **kwargs)
    return function_with_checking


@weight_checked
def weighted_quantile(points, weights=None, alpha=0.5):
    """
    Compute the weighted alpha-quantile. E.g. alpha = 0.5 -> median.
    """

    # sort input and set weights
    sorted_indices = np.argsort(points)
    points = points[sorted_indices]
    if weights is None:
        len_points = len(points)
        weights = np.ones(len_points) / len_points
    else:
        weights = weights[sorted_indices]

    cs = np.cumsum(weights)
    quantile = np.interp(alpha, cs - 0.5*weights, points)
    return quantile


@weight_checked
def weighted_median(points, weights):
    """
    Compute the weighted median (i.e. 0.5 quantile).
    """
    return weighted_quantile(points, weights, alpha=0.5)


@weight_checked
def weighted_mean(points, weights):
    """
    Compute the weighted mean.
    """
    return (points * weights).sum()


@weight_checked
def weighted_std(points, weights):
    """
    Compute the weighted standard deviation from the
    weighted mean.
    """
    mean = weighted_mean(points, weights)
    std = np.sqrt(((points - mean)**2 * weights).sum())
    return std


def effective_sample_size(weights):
    """
    Compute the effective sample size of weighted points
    sampled via importance sampling according to the formula

    .. math::
        n_\\text{eff} = \\frac{(\\sum_{i=1}^nw_i)^2}{\\sum_{i=1}^nw_i^2}
    """
    weights = np.array(weights)
    n_eff = np.sum(weights)**2 / np.sum(weights**2)
    return n_eff


def resample(points, weights, n):
    """
    Resample from weighted samples.

    Parameters
    ----------
    points:
        The random samples.
    weights:
        Weights of each sample point.
    n:
        Number of samples to resample.

    Returns
    -------
    resampled:
        A total of `n` points sampled from `points` with putting back
        according to `weights`.
    """
    weights = np.array(weights)
    weights /= np.sum(weights)
    resampled = np.random.choice(points, size=n, p=weights)
    return resampled


def resample_deterministic(points, weights, n, enforce_n=False):
    """
    Resample from weighted samples in a deterministic manner. Essentially,
    multiplicities are picked as follows:
    The weights are multiplied by the target number `n` and rounded to the
    nearest integer, potentially with correction if `enforce_n`.

    Parameters
    ----------
    points:
        The random samples.
    weights:
        Weights of each sample point.
    n:
        Number of samples to resample.
    enforce_n:
        Whether to enforce the returned array to have length `n`.
        If not, its length can be slightly off, but it may be more
        representative.

    Returns
    -------
    resampled:
        A total of (roughly) `n` points resampled from `points`
        deterministically using a rational representation of the `weights`.
    """
    weights = np.array(weights)
    numbers_f = weights * (n / np.sum(weights))

    numbers = np.round(numbers_f)

    # enforce return array length
    if enforce_n and np.sum(numbers) != n:
        residuals = numbers_f - numbers
        # sort the residuals mon. inc.
        order = np.argsort(residuals)
        # increment numbers with largest offsets
        while np.sum(numbers) < n:
            numbers[order[-1]] += 1
            order = order[:-1]
        # decrement numbers with largest negative offsets
        while np.sum(numbers) > n:
            numbers[order[0]] -= 1
            order = order[1:]

    resampled = []
    for i, ni in enumerate(numbers):
        resampled.extend([points[i]] * int(ni))

    return resampled
