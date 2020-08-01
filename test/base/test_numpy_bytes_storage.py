import pytest
import numpy as np
from pyabc.storage.numpy_bytes_storage import np_from_bytes, np_to_bytes


@pytest.fixture
def rand_arr():
    arr = np.random.rand(3, 8)
    return arr


def test_storage(rand_arr):
    arr = np_from_bytes(np_to_bytes(rand_arr))
    assert (arr == rand_arr).all()
