import scipy as sp


def weighted_quantile(points, weights=None, alpha=0.5):
    """
    Weighted alpha-quantile. E.g. alpha = 2 -> median.
    """

    # sort input and set weights
    sorted_indices = sp.argsort(points)
    points = points[sorted_indices]
    if weights is None:
        len_points = len(points)
        weights = sp.ones(len_points) / len_points
    else:
        weights = weights[sorted_indices]
    assert abs(weights.sum() - 1) < 1e-5, \
        ("Weights not normalized", weights.sum())

    cs = sp.cumsum(weights)
    quantile = sp.interp(alpha, cs - (1-alpha)*weights, points)
    return quantile


def weighted_median(points, weights):
    return weighted_quantile(points, weights, alpha=0.5)


def weighted_mean(points, weights):
    assert abs(weights.sum() - 1) < 1e-5, \
        ("Weights not normalized", weights.sum())

    return (points * weights).sum()


def weighted_std(points, weights):
    mean = weighted_mean(points, weights)
    std = sp.sqrt(((points - mean)**2 * weights).sum())
    return std
