"""Check pyabc.random_choice"""

import numpy as np
from time import time
from pyabc.random_choice import fast_random_choice


def test_fast_random_choice():
    """Check that `fast_random_choice` delivers the promised benefit."""
    nrep = 100000

    ws = np.random.uniform(size=(nrep, 10))
    ws /= ws.sum(axis=1, keepdims=True)

    np.random.seed(0)
    start = time()
    for w in ws:
        np.random.choice(len(w), p=w)
    time_numpy = time() - start
    print(f"Time numpy: {time_numpy}")

    np.random.seed(0)
    start = time()
    for w in ws:
        fast_random_choice(w)
    time_fast = time() - start
    print(f"Time fast_random_choice: {time_fast}")

    assert time_fast < time_numpy

    # for n >> 15, a small overhead of fast_random_choice persists,
    #  but whatever
