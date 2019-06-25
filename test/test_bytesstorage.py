import pytest
from pyabc.storage.bytes_storage import to_bytes, from_bytes
import pandas as pd
import numpy as np
import scipy as sp
from rpy2.robjects import r
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
import pyabc
import tempfile
import os


@pytest.fixture(params=["empty", "int", "float", "non_numeric_str",
                        "numeric_str", "int-float-numeric_str",
                        "int-float-non_numeric_str-str_ind",
                        "int-float-numeric_str-str_ind",
                        "py-int",
                        "py-float",
                        "py-str",
                        "r-df-cars",
                        "r-df-faithful"  # TODO re-add iris, see #45
                        ])
def object_(request):
    par = request.param
    if par == "empty":
        return pd.DataFrame()
    if par == "int":
        return pd.DataFrame({"a": sp.random.randint(-20, 20, 100),
                             "b": sp.random.randint(-20, 20, 100)})
    if par == "float":
        return pd.DataFrame({"a": sp.randn(100),
                             "b": sp.randn(100)})
    if par == "non_numeric_str":
        return pd.DataFrame({"a": ["foo", "bar"],
                             "b": ["bar", "foo"]})

    if par == "numeric_str":
        return pd.DataFrame({"a": list(map(str, sp.randn(100))),
                             "b": list(map(str,
                                           sp.random.randint(-20, 20, 100)))})
    if par == "int-float-numeric_str":
        return pd.DataFrame({"a": sp.random.randint(-20, 20, 100),
                             "b": sp.randn(100),
                             "c": list(map(str,
                                           sp.random.randint(-20, 20, 100)))})
    if par == "int-float-non_numeric_str-str_ind":
        return pd.DataFrame({"a": [1, 2],
                             "b": [1.1, 2.2],
                             "c": ["foo", "bar"]},
                            index=["first", "second"])
    if par == "int-float-numeric_str-str_ind":
        return pd.DataFrame({"a": [1, 2],
                             "b": [1.1, 2.2],
                             "c": ["1", "2"]},
                            index=["first", "second"])
    if par == "py-int":
        return 42
    if par == "py-float":
        return 42.42
    if par == "py-str":
        return "foo bar"
    if par == "np-int":
        return sp.random.randint(-20, 20, 100)
    if par == "np-float":
        return sp.random.randn(100)
    if par == "r-df-cars":
        return r["mtcars"]
    if par == "r-df-iris":
        return r["iris"]
    if par == "r-df-faithful":
        return r["faithful"]
    raise Exception("Invalid Test DataFrame Type")


def test_storage(object_):
    serial = to_bytes(object_)
    assert isinstance(serial, bytes)
    rebuilt = from_bytes(serial)

    if not isinstance(object_, robjects.DataFrame):
        assert isinstance(object_, type(rebuilt))

    if isinstance(object_, int):
        assert object_ == rebuilt
    elif isinstance(object_, float):
        assert object_ == rebuilt
    elif isinstance(object_, str):
        assert object_ == rebuilt
    elif isinstance(object_, np.ndarray):
        assert (object_ == rebuilt).all()
    elif isinstance(object_, pd.DataFrame):
        assert (object_ == rebuilt).all().all()
    elif isinstance(object_, robjects.DataFrame):
        assert (pandas2ri.ri2py(object_) == rebuilt).all().all()
    else:
        raise Exception("Could not compare")

def test_reference_parameter(object_):
    def model(parameter):
        return {"data": parameter["mean"] + 0.5 * sp.randn()}

    prior = pyabc.Distribution(mean=pyabc.RV("uniform", 0, 5))
    def distance(x, y):
        return abs(x["data"] - y["data"])

    abc = pyabc.ABCSMC(model, prior, distance, population_size=2)
    db_path = ("sqlite:///" +
               os.path.join(tempfile.gettempdir(), "test22.db"))
    observation = 2.5
    gt_parpp = {"theta": 1}
    abc.new(db_path, {"data": observation}, gt_par=gt_parpp)
    history = abc.run(minimum_epsilon=.01, max_nr_populations=1)
    a = pyabc.storage.history.History.get_parameter_reference_value(history, 'theta')
    if a[next(iter(a))] == gt_parpp['theta']:
        print('result are correct')