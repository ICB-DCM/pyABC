"""Transform dictionaries to arrays."""

from numbers import Number
from typing import List, Union

import numpy as np
import pandas as pd


def dict2arr(dct: Union[dict, np.ndarray], keys: List) -> np.ndarray:
    """Convert dictionary to 1d array, in specified key order.

    Parameters
    ----------
    dct: If dict-similar, values of all keys are extracted into a 1d array.
         Entries can be data frames, ndarrays, or single numbers.
    keys: Keys of interest, also defines the order.

    Returns
    -------
    arr: 1d array of all concatenated values.
    """
    if isinstance(dct, np.ndarray):
        return dct

    arr = []
    for key in keys:
        val = dct[key]
        if isinstance(val, (pd.DataFrame, pd.Series)):
            arr.append(val.to_numpy().flatten())
        elif isinstance(val, np.ndarray):
            arr.append(val.flatten())
        elif isinstance(val, Number):
            arr.append([val])
        else:
            raise TypeError(
                f"Cannot parse variable {key}={val} of type {type(val)} "
                "to numeric."
            )

    # for efficiency, directly parse single entries
    if len(arr) == 1:
        return np.asarray(arr[0])
    # flatten
    arr = [val for sub_arr in arr for val in sub_arr]
    return np.asarray(arr)


def dict2arrlabels(dct: dict, keys: List) -> List[str]:
    """Get label array consistent with the output of `dict2arr`.

    Can be called e.g. once on the observed data and used for logging.

    Parameters
    ----------
    dct: Model output or observed data.
    keys: Keys of interest, also defines the order.

    Returns
    -------
    labels: List of labels consistent with the output of `dict2arr`.
    """
    labels = []
    for key in keys:
        val = dct[key]
        if isinstance(val, (pd.DataFrame, pd.Series)):
            # default flattening mode is 'C', i.e. row-major, i.e. row-by-row
            for row in range(len(val.index)):
                for col in val.columns:
                    labels.append(f"{key}:{col}:{row}")
        elif isinstance(val, np.ndarray):
            # array can have any dimension, thus just flat indices
            for ix in range(val.size):
                labels.append(f"{key}:{ix}")
        elif isinstance(val, Number):
            labels.append(key)
        else:
            raise TypeError(
                f"Cannot parse variable {key}={val} of type {type(val)} "
                "to numeric."
            )
    return labels


def io_dict2arr(fun):
    """Wrapper parsing inputs dicts to ndarrays.

    Assumes the array is the first argument, and `self` holds a `keys`
    variable.
    """

    def wrapped_fun(self, data: Union[dict, np.ndarray], *args, **kwargs):
        # convert input to array
        data = dict2arr(data, self.x_keys)
        # call the actual function
        ret: np.ndarray = fun(self, data, *args, **kwargs)
        # flatten output
        return ret.flatten()

    return wrapped_fun
