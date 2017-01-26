from io import BytesIO
import numpy as np


def np_to_bytes(arr):
    """
    Serialize numpy array to bytes

    Parameters
    ----------
    arr: anything numpy.save with allow_pickle=False can store

    """
    f = BytesIO()
    np.save(f, arr, allow_pickle=False)
    f.seek(0)
    arr_bytes = f.read()
    f.close()
    return arr_bytes


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
    return arr
