import numpy as np
import pytest

from pyabc.random_choice import fast_random_choice


def test_fast_random_choice_basic():
    """Test the fast random choice function for various inputs."""
    # run with many values (method 1)
    weights = np.random.uniform(0, 1, size=10000)
    weights /= weights.sum()
    fast_random_choice(weights)

    # run with few values (method 2)
    weights = np.random.uniform(0, 1, size=5)
    weights /= weights.sum()
    fast_random_choice(weights)

    # run with a single value
    fast_random_choice(np.array([1]))


def test_fast_random_choice_errors():
    """Test the fast random choice function for invalid inputs."""
    with pytest.raises(ValueError):
        fast_random_choice(np.array([]))

    # non-normalized weights for many values
    weights = np.random.uniform(0, 1, size=1000)
    weights /= weights.sum() / 2
    with pytest.raises(ValueError):
        fast_random_choice(weights)

    # non-normalized weights for few values
    with pytest.raises(ValueError):
        # does not always raise, but will sometimes
        for _ in range(100):
            weights = np.random.uniform(0, 1, size=5)
            weights /= 5
            fast_random_choice(weights)


def test_fast_random_choice_output():
    """Test that the fast random choice outputs are accurate."""
    n_sample = 10000

    # few samples
    weights = np.array([0.1, 0.9])
    ret = np.array([fast_random_choice(weights) for _ in range(n_sample)])
    assert 0.08 < sum(ret == 0) / n_sample < 0.12
    assert 0.88 < sum(ret == 1) / n_sample < 0.92

    # many samples
    weights = np.array([*[0.02] * 20, *[0.3] * 2])
    ret = np.array([fast_random_choice(weights) for _ in range(n_sample)])
    assert 0.005 < sum(ret == 0) / n_sample < 0.04
    assert 0.28 < sum(ret == 20) / n_sample < 0.32
