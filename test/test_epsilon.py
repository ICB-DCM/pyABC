import pyabc
import numpy as np
import pandas as pd
import pytest


def test_noepsilon():
    eps = pyabc.NoEpsilon()
    assert not np.isfinite(eps(42))


def test_constantepsilon():
    eps = pyabc.ConstantEpsilon(42)
    assert np.isclose(eps(100), 42)


def test_listepsilon():
    eps = pyabc.ListEpsilon([3.5, 2.3, 1, 0.3])
    with pytest.raises(Exception):
        eps(4)


def test_quantileepsilon():
    mpl = 1.1
    df = pd.DataFrame({
        'distance': [1, 2, 3, 4],
        'w': [2, 1, 1, 1]
    })

    eps = pyabc.QuantileEpsilon(
        initial_epsilon=5.1, alpha=0.5,
        quantile_multiplier=mpl, weighted=False)

    # check if initial value is respected
    eps.initialize(0, lambda: df)
    assert np.isclose(eps(0), 5.1)

    # check if quantile is computed correctly
    eps.update(1, df)
    assert np.isclose(eps(1), mpl * 2.5)

    # use other quantile
    eps = pyabc.QuantileEpsilon(alpha=0.9, weighted=True)
    eps.initialize(0, lambda: df)
    assert eps(0) >= 3 and eps(0) <= 4


def test_medianepsilon():
    eps = pyabc.MedianEpsilon()
    assert np.isclose(eps.alpha, 0.5)
