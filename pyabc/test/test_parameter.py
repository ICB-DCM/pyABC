from pyabc import Parameter


def test_param_access():
    p = Parameter(a=1, b=2)
    assert p.a == 1
    assert p["a"] == 1


def test_param_access_from_dict():
    p = Parameter({"a": 1, "b": 2})
    assert p.a == 1
    assert p["a"] == 1