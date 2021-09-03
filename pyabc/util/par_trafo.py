"""Parameter transformations."""

import numpy as np
from typing import Callable, List, Union
from abc import ABC, abstractmethod

from .dict2arr import dict2arr


class ParTrafoBase(ABC):
    """Parameter transformations to use as regression targets.

    It may be useful to use as regression targets not simply the original
    parameters theta, but transformations thereof, such as moments
    theta**2.
    In particular, this can help overcome non-identifiabilities.
    """

    def initialize(self, keys: List[str]):
        """Initialize. Called once per analysis."""

    @abstractmethod
    def __call__(self, par_dict: dict) -> np.ndarray:
        """Transform parameters from input dict."""

    @abstractmethod
    def __len__(self):
        """Length of expected parameter transformation."""

    @abstractmethod
    def get_ids(self) -> List[str]:
        """Identifiers for the parameter transformations."""


class ParTrafo(ParTrafoBase):
    """Simple parameter transformation that accepts a list of transformations.

    The implementation assumes that each transformation maps n_par -> n_par.

    Parameters
    ----------
    trafos: Transformations to apply. Defaults to a single identity mapping.
    """

    def __init__(
        self,
        trafos: List[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.trafos = trafos

        # to maintain key order
        self.keys: Union[List[str], None] = None

    def initialize(self, keys: List[str]):
        # remember key order
        self.keys = keys

    def __call__(self, par_dict: dict) -> np.ndarray:
        # remember key order in case it was not set yet
        #  (fallback, uniqueness not guaranteed)
        if self.keys is None:
            self.keys = list(par_dict.keys())

        # to array
        out = dict2arr(par_dict, keys=self.keys)

        # apply transformations
        if self.trafos is not None:
            out = np.concatenate(
                [trafo(out).flatten() for trafo in self.trafos],
            )

        return out

    def __len__(self):
        if self.trafos is None:
            return len(self.keys)
        return len(self.keys) * len(self.trafos)

    def get_ids(self) -> List[str]:
        """
        Calculate keys as:
        {par_id_1}_{trafo_1}, ..., {par_id_n}_{trafo_1}, ...,
        {par_id_1}_{trafo_m}, ..., {par_id_n}_{trafo_m}
        """
        base_ids = [f"{key}" for key in self.keys]
        if self.trafos is None:
            return base_ids

        ids = [
            f"{base_id}_{trafo_ix}"
            for trafo_ix in range(len(self.trafos))
            for base_id in base_ids
        ]
        return ids
