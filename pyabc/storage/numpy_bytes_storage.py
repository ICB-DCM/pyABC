from io import BytesIO
import numpy as np


def np_to_bytes(arr):
    """
    Serialize numpy array to bytes

    Parameters
    ----------
    arr: anything numpy.save with allow_pickle=False can store


    .. note::

        Allowing for pickling is considered to be too dangerous.
        It is not guaranteed that pickled objects can be read back in later.

    """
    f = BytesIO()
    np.save(f, arr, allow_pickle=False)
    f.seek(0)
    arr_bytes = f.read()
    f.close()
    return arr_bytes


_primitive_types = [int, float, str]


def np_from_bytes(arr_bytes):
    """
    Load numpy array from bytes

    Parameters
    ----------
    arr_bytes: bytes as written to a file from np.save

    Returns
    -------
    arr: the deserialized array
    """
    f = BytesIO()
    f.write(arr_bytes)
    f.seek(0)
    arr = np.load(f)
    f.close()
    for type_ in _primitive_types:
        try:
            if type_(arr) == arr:
                return type_(arr)
        except (TypeError, ValueError):
            pass
    return arr
