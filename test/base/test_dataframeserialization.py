import pytest
from pyabc.storage.dataframe_bytes_storage import df_to_bytes, df_from_bytes
import pandas as pd
import numpy as np


@pytest.fixture(params=["empty", "int", "float", "non_numeric_str",
                        "numeric_str", "int-float-numeric_str",
                        "int-float-non_numeric_str-str_ind",
                        "int-float-numeric_str-str_ind"])
def df(request):
    par = request.param
    if par == "empty":
        return pd.DataFrame()
    if par == "int":
        return pd.DataFrame({"a": np.random.randint(-20, 20, 100),
                             "b": np.random.randint(-20, 20, 100)})
    if par == "float":
        return pd.DataFrame({"a": np.random.randn(100),
                             "b": np.random.randn(100)})
    if par == "non_numeric_str":
        return pd.DataFrame({"a": ["foo", "bar"],
                             "b": ["bar", "foo"]})

    if par == "numeric_str":
        return pd.DataFrame({"a": list(map(str, np.random.randn(100))),
                             "b": list(map(str,
                                           np.random.randint(-20, 20, 100)))})
    if par == "int-float-numeric_str":
        return pd.DataFrame({"a": np.random.randint(-20, 20, 100),
                             "b": np.random.randn(100),
                             "c": list(map(str,
                                           np.random.randint(-20, 20, 100)))})
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
    raise Exception("Invalid Test DataFrame Type")


def test_serialize(df):
    serial = df_to_bytes(df)
    assert isinstance(serial, bytes)
    rebuilt = df_from_bytes(serial)
    assert (df == rebuilt).all().all()
