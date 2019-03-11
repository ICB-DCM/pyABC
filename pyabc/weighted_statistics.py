import scipy as sp
import numpy as np


def weight_checked(function):
    """
    Function decorator to check normalization of weights.
    """
    def function_with_checking(points, weights=None, **kwargs):
        if weights is not None and not np.isclose(weights.sum(), 1):
            raise AssertionError(
                f"Weights not normalized: {weights.sum()}.")
        return function(points, weights, **kwargs)
    return function_with_checking


@weight_checked
def weighted_quantile(points, weights=None, alpha=0.5):
    """
    Weighted alpha-quantile. E.g. alpha = 0.5 -> median.
    """

    # sort input and set weights
    sorted_indices = sp.argsort(points)
    points = points[sorted_indices]
    if weights is None:
        len_points = len(points)
        weights = sp.ones(len_points) / len_points
    else:
        weights = weights[sorted_indices]

    cs = sp.cumsum(weights)
    quantile = sp.interp(alpha, cs - 0.5*weights, points)
    return quantile


@weight_checked
def weighted_median(points, weights):
    return weighted_quantile(points, weights, alpha=0.5)


@weight_checked
def weighted_mean(points, weights):
    return (points * weights).sum()


@weight_checked
def weighted_std(points, weights):
    mean = weighted_mean(points, weights)
    std = sp.sqrt(((points - mean)**2 * weights).sum())
    return std
