import pandas as pd

from .dataframe_bytes_storage import df_from_bytes, df_to_bytes
from .numpy_bytes_storage import np_from_bytes, np_to_bytes

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter

    def r_to_py(object_):
        if isinstance(object_, robjects.DataFrame):
            with localconverter(pandas2ri.converter):
                py_object_ = robjects.conversion.rpy2py(object_)
            return py_object_
        return object_

except ImportError:

    def r_to_py(object_):
        return object_


def maybe_to_df(object_):
    """
    Convert pd.Series and robjects.DataFrame to pd.DataFrame.
    """
    if isinstance(object_, pd.Series):
        object_ = object_.to_frame()
    object_ = r_to_py(object_)
    return object_


def to_bytes(object_):
    object_ = maybe_to_df(object_)
    if isinstance(object_, pd.DataFrame):
        return df_to_bytes(object_)
    return np_to_bytes(object_)


def from_bytes(bytes_):
    if bytes_[:6] == b"\x93NUMPY":
        return np_from_bytes(bytes_)
    return df_from_bytes(bytes_)
