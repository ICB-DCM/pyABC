import copy
import json

import numpy as np


def save_dict_to_json(dct: dict, file_: str):
    """
    Save dict to file. Inverse to `load_dict_from_json`.

    Parameters
    ----------
    dct:
        The dictionary to write to file.
    file_:
        Name of the file to write to.
    """
    dct = copy.deepcopy(dct)
    for key, val in dct.items():
        # cannot handle ndarrays
        if isinstance(val, np.ndarray):
            dct[key] = list(val)
    with open(file_, 'w') as f:
        json.dump(dct, f)


def load_dict_from_json(file_: str, key_type: type = int):
    """
    Read in json file. Convert keys to `key_type'.
    Inverse to `save_dict_to_json`.

    Parameters
    ----------
    file_:
        Name of the file to read in.
    key_type:
        Type to convert the keys into.

    Returns
    -------
    dct: The json file contents.
    """
    with open(file_, 'r') as f:
        _dct = json.load(f)
    dct = {}
    for key, val in _dct.items():
        dct[key_type(key)] = val
    return dct
