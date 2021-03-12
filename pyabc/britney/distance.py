import numpy as np
import pandas as pd
import copy
from abc import ABC, abstractmethod
from typing import Callable, Union

from ..population import Sample
from ..distance import Distance

from .scale import root_mean_square_deviation
from .util import dict2arr, io_dict2arr
from .sumstat import Sumstat, IdentitySumstat


class PNormDistance(Distance):
    """

    TODO
    * weights dict only makes sense if sumstat = IdentitySumstat with
      no modifications
    * same for factors
    * propagating input weights through trafos does not make much sense
    * allow passing initial weights (--> no preeq
    * ndarray (initial) weights and factors allowed with any sumstat, in that
      case just used

    * allow defining which weights should be shared? --> group by indicator,
      flatten, then compute weights separately, then fill weights into long arr
    """

    def __init__(
            self,
            sumstat: Sumstat = None,
            p: float = 2,
            weights: Union[np.ndarray, dict] = None,
            factors: Union[np.ndarray, dict] = None):
        super().__init__()
        if sumstat is None:
            sumstat = IdentitySumstat()
        self.p = p
        self.sumstat: Sumstat = sumstat
        self.weights = weights
        self.factors = factors

        # to cache the observed summary statistics
        self.x_0: Union[dict, None] = None
        self.s_0: Union[np.ndarray, None] = None

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        s, s0 = self.sumstat(x), self.sumstat(x_0)
        distances = np.abs(self.factors * self.weights * (s - s0))
        if self.p == np.inf:
            return distances.max()
        return np.sum(distances**self.p)

    def initialize(
            self,
            t: int,
            get_sample: Callable[[], Sample],
            x_0: dict = None):
        self.sumstat.initialize(t=t, get_sample=get_sample, x_0=x_0)

        # observed data
        self.x_0 = x_0
        self.s_0 = self.sumstat(x_0)
        if self.weights is None:
            self.weights = np.ones(len(self.s_0))
        elif isinstance(self.weights, dict):
            self.weights = dict2arr(self.weights, self.sumstat.get_labels())
        if self.factors is None:
            self.factors = np.ones(len(self.s_0))
        elif isinstance(self.factors, dict):
            self.factors = dict2arr(self.factors, self.sumstat.get_labels())

    def update(
            self,
            t: int,
            get_sample: Callable[[], Sample]) -> bool:
        updated = self.sumstat.update(t, get_sample)
        if updated:
            self.s_0 = self.sumstat(self.x_0)
        return updated


class AdaptivePNormDistance(PNormDistance):

    def __init__(self, sumstat, adaptive: bool = False, scale_function: Callable = None):
        super().__init__(sumstat)
        self.adaptive = adaptive
        if scale_function is None:
            scale_function = root_mean_square_deviation
        self.scale_function = scale_function

    def initialize(
            self,
            t: int,
            get_sample: Callable[[], Sample],
            x_0: dict = None):
        super().initialize(t, get_sample, x_0)
        sample = get_sample()
        self._fit(sample)

    def _fit(self, sample: Sample):
        particles = sample.all_particles

        ss = np.array([self.britney(p.sum_stat).flatten() for p in particles])

        s0 = self.britney(self.britney.x0)
        scales = self.scale_function(data=ss, x_0=s0)
        self.weights = 1 / scales

    def update(
            self,
            t: int,
            get_sample: Callable[[], Sample]) -> bool:
        updated = super().update(t, get_sample)
        if not self.adaptive:
            return updated
        sample = get_sample()
        self._fit(sample)
        return True


class GimmeMoreEuclidean(Euclidean):
    """Weight by something like |dfdy_j| / (std(y_j) + |bias(y_j)|)"""

    def __init__(
            self, sumstat: Britney, adaptive: bool = True,
            scale_function: Callable = None):
        super().__init__(sumstat)
        self.adaptive = adaptive
        if scale_function is None:
            scale_function = root_mean_square_deviation
        self.scale_function = scale_function

    def initialize(
            self,
            t: int,
            get_sample: Callable[[], Sample],
            x_0: dict = None):
        super().initialize(t, get_sample, x_0)
        sample = get_sample()
        self._fit(sample)

    def _fit(self, sample: Sample):
        particles = sample.accepted_particles

        xs = np.array([dict2arr(p.sum_stat).flatten() for p in particles])
        x0 = self.britney.x0

        # normalize to unit scale (variance and bias)
        scale = self.scale_function(data=xs, x_0=x0)

        # calculate finite differences

