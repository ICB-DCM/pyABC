from .numpy_bytes_storage import np_from_bytes, np_to_bytes
from .dataframe_bytes_storage import df_to_bytes, df_from_bytes
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri


def to_bytes(object_):
    if isinstance(object_, robjects.DataFrame):
        object_ = pandas2ri.ri2py(object_).copy()
    if isinstance(object_, pd.DataFrame):
        return df_to_bytes(object_)
    return np_to_bytes(object_)


def from_bytes(bytes_):
    if bytes_[:6] == b"\x93NUMPY":
        return np_from_bytes(bytes_)
    return df_from_bytes(bytes_)
