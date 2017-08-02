import json
from numbers import Number
import pandas as pd


def maybe_to_json(x):
    """

    Parameters
    ----------
    x: np.ndarray, pd.DataFrame, int, float, str

    Returns
    -------

      * JSON representation of x if x is a np.ndarray or pd.DataFrame
      * x itself if x is str, int or float

    """
    try:
        return x.to_json(orient="records")
    except AttributeError:
        pass
    try:
        return json.dumps(x.tolist())
    except AttributeError:
        pass
    if isinstance(x, Number):
        return x
    if isinstance(x, str):
        return x
    return json.dumps(x)


def sumstat_to_json(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df:
        if c.startswith("sumstat"):
            df[c] = df[c].map(maybe_to_json)
    return df


def to_file(df: pd.DataFrame, file: str, file_format="feather"):
    df_json = sumstat_to_json(df)
    df_json_no_index = df_json.reset_index()
    getattr(df_json_no_index, "to_" + file_format)(file)
