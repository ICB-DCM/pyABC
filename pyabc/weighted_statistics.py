import scipy as sp


def weight_checked(function):
    """
    Function decorator to check normalization of weights.
    """
    def function_with_checking(points, weights):
        assert abs(weights.sum() - 1) < 1e-5, \
            ("Weights not normalized", weights.sum())
        return function(points, weights)
    return function_with_checking


@weight_checked
def weighted_median(points, weights):
        sorted_indices = sp.argsort(points)
        points = points[sorted_indices]
        weights = weights[sorted_indices]
        cs = sp.cumsum(weights)
        median = sp.interp(.5, cs - .5*weights, points)
        return median


@weight_checked
def weighted_mean(points, weights):
    return (points * weights).sum()


@weight_checked
def weighted_std(points, weights):
    mean = weighted_mean(points, weights)
    std = sp.sqrt(((points - mean)**2 * weights).sum())
    return std
