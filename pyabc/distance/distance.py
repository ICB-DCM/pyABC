import scipy as sp
import numpy as np
from scipy import linalg as la
from typing import List, Callable
import logging

from ..sampler import Sampler
from .scales import standard_deviation
from .base import Distance


logger = logging.getLogger("Distance")


class PNormDistance(Distance):
    """
    Use weighted p-norm

    .. math::

        d(x, y) = \
        \\left [\\sum_{i} \\left| w_i ( x_i-y_i ) \\right|^{p} \\right ]^{1/p}

    to compute distances between sets of summary statistics. E.g. set p=2 to
    get a Euclidean distance.

    Parameters
    ----------

    p: float
        p for p-norm. Required p >= 1, p = np.inf allowed (infinity-norm).

    w: dict
        Weights. Dictionary indexed by time points. Each entry contains a
        dictionary of numeric weights, indexed by summary statistics labels.
        If None is passed, a weight of 1 is considered for every summary
        statistic. If no entry is available in w for a given time point,
        the maximum available time point is selected.
    """

    def __init__(self,
                 p: float = 2,
                 w: dict = None):
        super().__init__()

        if p < 1:
            raise ValueError("It must be p >= 1")
        self.p = p

        self.w = w

    def __call__(self,
                 x: dict,
                 x_0: dict,
                 t: int,
                 par: dict = None) -> float:
        # make sure weights are initialized
        if self.w is None:
            self._set_default_weights(t, x.keys())

        # select last time point for which weights exist
        if t not in self.w:
            t = max(self.w)

        # extract weights for time point
        w = self.w[t]

        # compute distance
        if self.p == np.inf:
            d = max(abs(w[key] * (x[key] - x_0[key]))
                    if key in x and key in x_0 else 0
                    for key in w)
        else:
            d = pow(
                sum(pow(abs(w[key] * (x[key] - x_0[key])), self.p)
                    if key in x and key in x_0 else 0
                    for key in w),
                1 / self.p)

        return d

    def _set_default_weights(self,
                             t: int,
                             sum_stat_keys):
        """
        Init weights to 1 for every summary statistic.
        """
        self.w = {t: {k: 1 for k in sum_stat_keys}}

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__,
                "p": self.p,
                "w": self.w}


class AdaptivePNormDistance(PNormDistance):
    """
    In the p-norm distance, adapt the weights for each generation, based on
    the previous simulations. This class is motivated by [#prangle]_.

    Parameters
    ----------

    p: float, optional (default = 2)
        p for p-norm. Required p >= 1, p = np.inf allowed (infinity-norm).

    adaptive: bool, optional (default = True)
        True: Adapt distance after each iteration.
        False: Adapt distance only once at the beginning in initialize().
        This corresponds to a pre-calibration.

    scale_function: Callable, optional (default = standard_deviation)
        (data: list, x_0: float) -> scale: float. Computes the scale (i.e.
        inverse weight s = 1 / w) for a given summary statistic. Here, data
        denotes the list of simulated summary statistics, and x_0 the observed
        summary statistic. Implemented are absolute_median_deviation,
        standard_deviation (default), centered_absolute_median_deviation,
        centered_standard_deviation.

    normalize_weights: bool, optional (default = True)
        Whether to normalize the weights to have mean 1. This just possibly
        smoothes the decrease of epsilon and might aid numeric stability, but
        is not strictly necessary.

    max_weight_ratio: float, optional (default = None)
        If not None, large weights will be bounded by the ratio times the
        smallest non-zero absolute weight. In practice usually not necessary,
        it is theoretically required to ensure convergence.


    .. [#prangle] Prangle, Dennis. "Adapting the ABC Distance Function".
                Bayesian Analysis, 2017. doi:10.1214/16-BA1002.
    """

    def __init__(self,
                 p: float = 2,
                 adaptive: bool = True,
                 scale_function=None,
                 normalize_weights: bool = True,
                 max_weight_ratio: float = None):
        # call p-norm constructor
        super().__init__(p=p, w=None)

        self.adaptive = adaptive

        if scale_function is None:
            scale_function = standard_deviation
        self.scale_function = scale_function

        self.normalize_weights = normalize_weights
        self.max_weight_ratio = max_weight_ratio

        self.x_0 = None

    def configure_sampler(self,
                          sampler: Sampler):
        """
        Make the sampler return also rejected particles,
        because these are needed to get a better estimate of the summary
        statistic variabilities, avoiding a bias to accepted ones only.

        Parameters
        ----------

        sampler: Sampler
            The sampler employed.
        """
        if self.adaptive:
            sampler.sample_factory.record_rejected = True

    def initialize(self,
                   t: int,
                   get_sum_stats: Callable[[], List[dict]],
                   x_0: dict = None):
        """
        Initialize weights.
        """
        self.x_0 = x_0

        # execute function
        sum_stats = get_sum_stats()

        # update weights from samples
        self._update(t, sum_stats)

    def update(self,
               t: int,
               sum_stats: List[dict]):
        """
        Update weights based on all simulations.
        """

        if not self.adaptive:
            return False

        self._update(t, sum_stats)

        return True

    def _update(self,
                t: int,
                sum_stats: List[dict]):
        """
        Here the real update of weights happens.
        """

        # retrieve keys
        keys = self.x_0.keys()

        # number of samples
        n_samples = len(sum_stats)

        # make sure w_list is initialized
        if self.w is None:
            self.w = {}

        # to-be-filled-and-appended weights dictionary
        w = {}

        for key in keys:
            # prepare list for key
            current_list = []
            for j in range(n_samples):
                if key in sum_stats[j]:
                    current_list.append(sum_stats[j][key])

            # compute scaling
            scale = self.scale_function(data=current_list, x_0=self.x_0[key])

            # compute weight (inverted scale)
            if np.isclose(scale, 0):
                # This means that either the summary statistic is not in the
                # samples, or that all simulations were identical. In either
                # case, it should be safe to ignore this summary statistic.
                w[key] = 0
            else:
                w[key] = 1 / scale

        # normalize weights to have mean 1
        w = self._normalize_weights(w)

        # bound weights
        w = self._bound_weights(w)

        # add to w attribute, at time t
        self.w[t] = w

        # logging
        logger.debug("update distance weights = {}".format(self.w[t]))

    def _normalize_weights(self, w):
        """
        Normalize weights to have mean 1.

        This has just the effect that eps will decrease more smoothly, but is
        not important otherwise.
        """
        if not self.normalize_weights:
            return w

        mean_weight = np.mean(list(w.values()))
        for key in w:
            w[key] /= mean_weight

        return w

    def _bound_weights(self, w):
        """
        Bound all weights to self.max_weight_ratio times the minimum
        non-zero absolute weight, if self.max_weight_ratio is not None.

        While this is usually not required in practice, it is theoretically
        necessary that the ellipses are not arbitrarily eccentric, in order
        to ensure convergence.
        """
        if self.max_weight_ratio is None:
            return w

        # find minimum weight != 0
        w_arr = np.array(list(w.values()))
        min_abs_weight = np.min(np.abs(w_arr[w_arr != 0]))
        # can be assumed to be != 0

        for key, value in w.items():
            # bound too large weights
            if abs(value) / min_abs_weight > self.max_weight_ratio:
                w[key] = np.sign(value) * self.max_weight_ratio \
                    * min_abs_weight

        return w

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__,
                "p": self.p,
                "adaptive": self.adaptive,
                "scale_function": self.scale_function.__name__,
                "normalize_weights": self.normalize_weights,
                "max_weight_ratio": self.max_weight_ratio}


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

    def __init__(self,
                 measures_to_use='all'):
        super().__init__()
        # the measures (summary statistics) to use for distance calculation
        self.measures_to_use = measures_to_use

    def initialize(self,
                   t: int,
                   get_sum_stats: Callable[[], List[dict]],
                   x_0: dict = None):
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

    def __call__(self,
                 x: dict,
                 x_0: dict,
                 t: int = None,
                 par: dict = None) -> float:
        return sum(abs((x[key] - x_0[key]) / x_0[key]) if x_0[key] != 0 else
                   (0 if x[key] == 0 else np.inf)
                   for key in self.measures_to_use) / len(self.measures_to_use)


class PCADistance(DistanceWithMeasureList):
    """
    Calculate distance in whitened coordinates.

    A whitening transformation :math:`X` is calculated from an initial sample.
    The distance is measured as euclidean distance in the transformed space.
    I.e

    .. math::

        d(x,y) = \\| Wx - Wy \\|
    """

    def __init__(self, measures_to_use='all'):
        super().__init__(measures_to_use)
        self._whitening_transformation_matrix = None

    def _dict_to_vect(self, x):
        return sp.asarray([x[key] for key in self.measures_to_use])

    def _calculate_whitening_transformation_matrix(self, sum_stats):
        samples_vec = sp.asarray([self._dict_to_vect(x)
                                  for x in sum_stats])
        # samples_vec is an array of shape nr_samples x nr_features
        means = samples_vec.mean(axis=0)
        centered = samples_vec - means
        covariance = centered.T.dot(centered)
        w, v = la.eigh(covariance)
        self._whitening_transformation_matrix = (
            v.dot(sp.diag(1. / sp.sqrt(w))).dot(v.T))

    def initialize(self,
                   t: int,
                   get_sum_stats: Callable[[], List[dict]],
                   x_0: dict = None):
        super().initialize(t, get_sum_stats, x_0)

        # execute function
        sum_stats = get_sum_stats()

        self._calculate_whitening_transformation_matrix(sum_stats)

    def __call__(self,
                 x: dict,
                 x_0: dict,
                 t: int = None,
                 par: dict = None) -> float:
        x_vec, x_0_vec = self._dict_to_vect(x), self._dict_to_vect(x_0)
        distance = la.norm(
            self._whitening_transformation_matrix.dot(x_vec - x_0_vec), 2)
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
        self.normalization = {measure:
                              self.upper(measures[measure])
                              - self.lower(measures[measure])
                              for measure in self.measures_to_use}

    def initialize(self,
                   t: int,
                   get_sum_stats: Callable[[], List[dict]],
                   x_0: dict = None):
        super().initialize(t, get_sum_stats, x_0)

        # execute function
        sum_stats = get_sum_stats()

        self._calculate_normalization(sum_stats)

    def __call__(self,
                 x: dict,
                 x_0: dict,
                 t: int = None,
                 par: dict = None) -> float:
        distance = sum(abs((x[key] - x_0[key]) / self.normalization[key])
                       for key in self.measures_to_use)
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
        return sp.percentile(parameter_list,
                             100 - PercentileDistance.PERCENTILE)

    @staticmethod
    def lower(parameter_list):
        return sp.percentile(parameter_list,
                             PercentileDistance.PERCENTILE)

    def get_config(self):
        config = super().get_config()
        config["PERCENTILE"] = self.PERCENTILE
        return config
