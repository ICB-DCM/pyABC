import numpy as np
from scipy.stats import ks_2samp
import pyabc.weighted_statistics as ws


points = np.array([1, 5, 2.5])
weights = np.array([0.5, 0.2, 0.3])


def test_weighted_quantile():
    q = ws.weighted_quantile(points, weights)
    assert 1 < q and q < 2.5
    q = ws.weighted_quantile(points, weights, alpha=0.2)
    assert q == 1
    q = ws.weighted_quantile(points, weights, alpha=0.8)
    assert 2.5 < q and q < 5
    q = ws.weighted_quantile(points, weights, alpha=0.9)
    assert q == 5
    q = ws.weighted_quantile(points, weights, alpha=1.0)
    assert q == 5


def test_weighted_median():
    m = ws.weighted_median(points, weights)
    assert 1 < m and m < 2.5


def test_weighted_mean():
    m = ws.weighted_mean(points, weights)
    assert m == 2.25


def test_weighted_std():
    std = ws.weighted_std(points, weights)

    m_ = np.sum(points * weights)
    std_ = np.sqrt(np.sum(weights * (points - m_)**2))

    assert std == std_


def test_resample():
    """
    Test that the resampling process yields consistent distributions,
    using a KS test.
    """
    nw = 50  # number of weighted points
    points = np.random.randn(nw)
    weights = np.random.rand(nw)
    weights /= np.sum(weights)

    n = 1000  # number of non-weighted points
    # sample twice from same samples
    resampled1 = ws.resample(points, weights, n)
    resampled2 = ws.resample(points, weights, n)

    # should be same distribution
    _, p = ks_2samp(resampled1, resampled2)
    assert p > 1e-2

    # use different points
    points3 = np.random.randn(nw)
    resampled3 = ws.resample(points3, weights, n)
    # should be different distributions
    _, p = ks_2samp(resampled1, resampled3)
    assert p < 1e-2


def test_resample_deterministic():
    """
    Test the deterministic resampling routine.
    """
    nw = 50  # number of weighed points
    points = np.random.randn(nw)
    weights = np.random.rand(nw)
    weights /= np.sum(weights)

    n = 1000  # number of non-weighted points
    resampled_det = ws.resample_deterministic(points, weights, n, False)

    resampled = ws.resample(points, weights, n)

    # should be same distribution
    _, p = ks_2samp(resampled_det, resampled)
    assert p > 1e-2

    resampled_det2 = ws.resample_deterministic(points, weights, n, True)
    assert len(resampled_det2) == n

    _, p = ks_2samp(resampled_det2, resampled)
    assert p > 1e-2
