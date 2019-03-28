import numpy as np

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
