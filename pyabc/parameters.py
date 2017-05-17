from collections import UserDict
from typing import List


class ParameterStructure(UserDict):
    @staticmethod
    def flatten_dict(dict_: dict):
        new_dict = {}
        for key, value in dict_.items():
            if isinstance(value, dict):
                flattened = ParameterStructure.flatten_dict(value)
                for key_flat, value_flat in flattened.items():
                    new_dict.update({str(key) + "." + key_flat: value_flat})
            else:
                new_dict.update({key: value})
        return new_dict

    def __init__(self, *args, **kwargs):
        if len(args) > 0 and len(kwargs) > 0:
            raise Exception("Only keyword or dictionary allowed")
        if len(args) > 0:
            flattened = ParameterStructure.flatten_dict(args[0])
        if len(kwargs) > 0:
            flattened = ParameterStructure.flatten_dict(kwargs)
        if len(args) == 0 and len(kwargs) == 0:
            flattened = {}
        super().__init__(flattened)


class Parameter(ParameterStructure):
    """
    A single model parameter.

    Parameters are essentially a dictionary with the additional functionality
    to add and subtract parameters.

    I.e. ``par_1 + par_2`` adds key wise.

    Contents can be accessed with square brackets or in dot notation.

    For example

    .. code:: python

        >>> p = Parameter(a=1, b=2)
        >>> assert p.a == p["a"]

    or

    .. code:: python

        >>> p = Parameter({"a": 1, "b": 2})
        >>> assert p.a == p["a"]

    """
    def __add__(self, other: "Parameter") -> "Parameter":
        return Parameter(**{key: self[key] + other[key] for key in self})

    def __sub__(self, other: "Parameter") -> "Parameter":
        return Parameter(**{key: self[key] - other[key] for key in self})

    def __repr__(self):
        return "<Parameter " + super().__repr__()[1:-1] + ">"

    def __getattr__(self, item):
        """
        Convenience for dot notation access.
        """
        try:
            return self[item]
        except KeyError:
            raise AttributeError

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.data = state

    def copy(self) -> "Parameter":
        """
        Copy the parameter.
        """
        return Parameter(**self)


class ValidParticle:
    """
    A valid (accepted) particle.

    Parameters
    ----------

    m: int
        The model nr

    parameter: Parameter
        The model specific parameter

    weight: float, 0 < weight < 1
        The weight of the particle

    distance_list: List[float]
        A particle can contain more than one sample.
        If so, the distances of the individual samples
        are stored in this list. In the most common case of a single
        sample, this list has length 1.

    summary_statistics_list
        List of summary statistics which describe the sample
        This list is usually of length 1. This list is longer only if more
        than one sample is taken for a particle.
    """
    def __init__(self, m: int, parameter: Parameter,
                 weight: float, distance_list: List[float],
                 summary_statistics_list: List[dict]):
        self.m = m
        self.parameter = parameter
        self.weight = weight
        self.distance_list = distance_list
        self.summary_statistics_list = summary_statistics_list

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __eq__(self, other):
        for key in ["m", "parameter", "weight", "distance_list",
                    "summary_statistics_list"]:
            if self[key] != other[key]:
                return False
        return True
