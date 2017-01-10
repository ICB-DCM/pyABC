from pyabc import Parameter
import pickle


def test_param_access():
    p = Parameter(a=1, b=2)
    assert p.a == 1
    assert p["a"] == 1


def test_param_access_from_dict():
    p = Parameter({"a": 1, "b": 2})
    assert p.a == 1
    assert p["a"] == 1


def test_pickle():
    p = Parameter({"a": 1, "b": 2})
    s = pickle.dumps(p)
    l = pickle.loads(s)
    assert l is not p
    assert l == p
