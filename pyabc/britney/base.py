import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List, Union


def dict_to_1d_arr(dct: dict, keys: List):
    """Convert dictionary to array, in specified key order."""
    arr = []
    for key in keys:
        val = dct[key]
        if isinstance(val, (pd.DataFrame, pd.Series)):
            arr.extend(val.to_numpy().flatten())
        elif isinstance(val, np.ndarray):
            arr.extend(val.flatten())
        else:
            arr.append(val)
    return np.asarray(arr)


def input_to_ndarray(fun):
    def wrapped_fun(self, d: Union[dict, np.ndarray]):
        if isinstance(d, dict):
            return dict_to_1d_arr(d, self.keys)
        return d
    return wrapped_fun()



class Britney(ABC):

    @abstractmethod
    def __call__(self, d: Union[dict, np.ndarray]):
        pass

    def update(self):
        pass


class IdentityBritney:

    @input_to_ndarray
    def __call__(self, d):
        return d


class PolynomialExpansionBritney:

    def __init__(self, degree: int, cross: bool = False):
        self.degree = degree
        self.cross = cross

    @input_to_ndarray
    def __call__(self, d):
        result =



class LearnedLinearBritney(Britney):

    pass




