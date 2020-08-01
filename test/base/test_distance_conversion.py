import pytest
from pyabc.distance import to_distance


def f(x, y):
    return 0


@pytest.fixture(params=[None, f, lambda x: x])
def distance(request):
    return request.param


def test_distance_none(distance):
    dist_func = to_distance(distance)
    config = dist_func.get_config()
    assert "name" in config
