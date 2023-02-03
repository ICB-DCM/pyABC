"""Various basic distances."""

import logging
from typing import Callable, List, Union

import numpy as np
from scipy import linalg as la

from ..population import Sample
from .base import Distance

logger = logging.getLogger("ABC.Distance")


class DistanceWithMeasureList(Distance):
    """
    Base class for distance functions with measure list.
    This class is not functional on its own.

    Parameters
    ----------
    measures_to_use: Union[str, List[str]].
        * If set to "all", all measures are used. This is the default.
        * If a list is provided, the measures in the list are used.
        * measures refers to the summary statistics.
    """

    def __init__(
        self,
        measures_to_use='all',
    ):
        super().__init__()
        # the measures (summary statistics) to use for distance calculation
        self.measures_to_use = measures_to_use

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ):
        if self.measures_to_use == 'all':
            self.measures_to_use = x_0.keys()

    def get_config(self):
        config = super().get_config()
        config["measures_to_use"] = self.measures_to_use
        return config


class ZScoreDistance(DistanceWithMeasureList):
    """
    Calculate distance as sum of ZScore over the selected measures.
    The measured Data is the reference for the ZScore.

    Hence

    .. math::
        d(x, y) = \
        \\sum_{i \\in \\text{measures}} \\left| \\frac{x_i-y_i}{y_i} \\right|
    """

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None,
    ) -> float:
        return sum(
            abs((x[key] - x_0[key]) / x_0[key])
            if x_0[key] != 0
            else (0 if x[key] == 0 else np.inf)
            for key in self.measures_to_use
        ) / len(self.measures_to_use)


class PCADistance(DistanceWithMeasureList):
    """
    Calculate distance in whitened coordinates.

    A PCA whitening transformation :math:`X` is calculated from an initial
    sample.
    The distance is measured as p-norm distance in the transformed space.
    I.e

    .. math::

        d(x,y) = \\| Wx - Wy \\|

    Parameters
    ----------
    measures_to_use: See DistanceWithMeasureList.
    p: p-norm, defaults to Euclidean distance.
    """

    def __init__(self, measures_to_use='all', p: float = 2):
        super().__init__(measures_to_use)
        self.p: float = p
        self.trafo: Union[np.ndarray, None] = None

    def _dict_to_vect(self, x):
        return np.asarray([x[key] for key in self.measures_to_use])

    def _calculate_whitening_transformation_matrix(self, sum_stats):
        # create data matrix, shape (n_sample, n_y)
        x = np.asarray([self._dict_to_vect(x) for x in sum_stats])
        # center
        mean = np.mean(x, axis=0)
        x -= mean
        # covariance matrix, with bias correction
        cov = (x.T @ x) / (x.shape[0] - 1)
        # eigenvalues and eigenvectors
        ew, ev = la.eigh(cov)
        # whitening transformation
        self.trafo = np.diag(1.0 / np.sqrt(ew)) @ ev.T

    def requires_calibration(self) -> bool:
        return True

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ):
        super().initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )

        # execute function
        all_sum_stats = get_sample().all_sum_stats

        self._calculate_whitening_transformation_matrix(all_sum_stats)

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None,
    ) -> float:
        x_vec, x_0_vec = self._dict_to_vect(x), self._dict_to_vect(x_0)
        distance = la.norm(
            self.trafo @ (x_vec - x_0_vec).reshape(-1, 1), ord=self.p
        )
        return distance


class RangeEstimatorDistance(DistanceWithMeasureList):
    """
    Abstract base class for distance functions which estimate is based on a
    range.

    It defines the two template methods ``lower`` and ``upper``.

    Hence

    .. math::

        d(x, y) = \
        \\sum_{i \\in \\text{measures}} \\left | \\frac{x_i - y_i}{u_i - l_i}\
          \\right |

    where :math:`l_i` and :math:`u_i` are the lower and upper
    margin for measure :math:`i`.
    """

    @staticmethod
    def lower(parameter_list: List[float]):
        """
        Calculate the lower margin form a list of parameter values.

        Parameters
        ----------
        parameter_list: List[float]
            List of values of a parameter.

        Returns
        -------
        lower_margin: float
            The lower margin of the range calculated from these parameters
        """

    @staticmethod
    def upper(parameter_list: List[float]):
        """
        Calculate the upper margin form a list of parameter values.

        Parameters
        ----------
        parameter_list: List[float]
            List of values of a parameter.

        Returns
        -------
        upper_margin: float
            The upper margin of the range calculated from these parameters
        """

    def __init__(self, measures_to_use='all'):
        super().__init__(measures_to_use)
        self.normalization = None

    def get_config(self):
        config = super().get_config()
        config["normalization"] = self.normalization
        return config

    def _calculate_normalization(self, sum_stats):
        measures = {name: [] for name in self.measures_to_use}
        for sample in sum_stats:
            for measure in self.measures_to_use:
                measures[measure].append(sample[measure])
        self.normalization = {
            measure: self.upper(measures[measure])
            - self.lower(measures[measure])
            for measure in self.measures_to_use
        }

    def requires_calibration(self) -> bool:
        return True

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ):
        super().initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )

        # execute function
        all_sum_stats = get_sample().all_sum_stats

        self._calculate_normalization(all_sum_stats)

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None,
    ) -> float:
        distance = sum(
            abs((x[key] - x_0[key]) / self.normalization[key])
            for key in self.measures_to_use
        )
        return distance


class MinMaxDistance(RangeEstimatorDistance):
    """
    Calculate upper and lower margins as max and min of the parameters.
    This works surprisingly well for normalization in simple cases
    """

    @staticmethod
    def upper(parameter_list):
        return max(parameter_list)

    @staticmethod
    def lower(parameter_list):
        return min(parameter_list)


class PercentileDistance(RangeEstimatorDistance):
    """
    Calculate normalization 20% and 80% from percentiles as lower
    and upper margins
    """

    PERCENTILE = 20  #: The percentiles

    @staticmethod
    def upper(parameter_list):
        return np.percentile(
            parameter_list, 100 - PercentileDistance.PERCENTILE
        )

    @staticmethod
    def lower(parameter_list):
        return np.percentile(parameter_list, PercentileDistance.PERCENTILE)

    def get_config(self):
        config = super().get_config()
        config["PERCENTILE"] = self.PERCENTILE
        return config
