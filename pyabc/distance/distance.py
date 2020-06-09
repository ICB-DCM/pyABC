"""Distance functions."""

import numpy as np
from scipy import linalg as la
from typing import List, Callable, Union
import logging

from ..sampler import Sampler
from .scale import standard_deviation, span
from .base import Distance, to_distance
from ..storage import save_dict_to_json


logger = logging.getLogger("Distance")


class PNormDistance(Distance):
    """
    Use a weighted p-norm

    .. math::

        d(x, y) = \
        \\left [\\sum_{i} \\left| w_i ( x_i-y_i ) \\right|^{p} \\right ]^{1/p}

    to compute distances between sets of summary statistics. E.g. set p=2 to
    get a Euclidean distance.

    Parameters
    ----------

    p: float, optional (default = 2)
        p for p-norm. Required p >= 1, p = np.inf allowed (infinity-norm).
    weights: dict, optional (default = 1)
        Weights. Dictionary indexed by time points. Each entry contains a
        dictionary of numeric weights, indexed by summary statistics labels.
        If None is passed, a weight of 1 is considered for every summary
        statistic. If no entry is available in `weights` for a given time
        point, the maximum available time point is selected.
        It is also possible to pass a single dictionary index by summary
        statistics labels, if weights do not change in time.
    factors: dict, optional (default = 1)
        Scaling factors that the weights are multiplied with. The same
        structure applies as to weights.
        If None is passed, a factor of 1 is considered for every summary
        statistic.
        Note that in this class, factors are superfluous as everything can
        be achieved with weights alone, however in subclasses the factors
        can remain static while weights adapt over time, allowing for
        greater flexibility.
    """

    def __init__(self,
                 p: float = 2,
                 weights: dict = None,
                 factors: dict = None):
        super().__init__()

        if p < 1:
            raise ValueError("It must be p >= 1")
        self.p = p

        self.weights = weights
        self.factors = factors

    def initialize(self,
                   t: int,
                   get_all_sum_stats: Callable[[], List[dict]],
                   x_0: dict = None):
        super().initialize(t, get_all_sum_stats, x_0)
        self.format_weights_and_factors(t, x_0.keys())

    def format_weights_and_factors(self, t, sum_stat_keys):
        self.weights = PNormDistance.format_dict(
            self.weights, t, sum_stat_keys)
        self.factors = PNormDistance.format_dict(
            self.factors, t, sum_stat_keys)

    def __call__(self,
                 x: dict,
                 x_0: dict,
                 t: int = None,
                 par: dict = None) -> float:
        # make sure everything is formatted correctly
        self.format_weights_and_factors(t, x_0.keys())

        # extract values for given time point
        w = PNormDistance.get_for_t_or_latest(self.weights, t)
        f = PNormDistance.get_for_t_or_latest(self.factors, t)

        # compute distance
        if self.p == np.inf:
            # maximum absolute distance
            d = max(abs((f[key] * w[key]) * (x[key] - x_0[key]))
                    if key in x and key in x_0 else 0
                    for key in w)
        else:
            # weighted p-norm distance
            d = pow(
                sum(pow(abs((f[key] * w[key]) * (x[key] - x_0[key])), self.p)
                    if key in x and key in x_0 else 0
                    for key in w),
                1 / self.p)

        return d

    def get_config(self) -> dict:
        return {"name": self.__class__.__name__,
                "p": self.p,
                "weights": self.weights,
                "factors": self.factors}

    @staticmethod
    def format_dict(w, t, sum_stat_keys, default_val=1.):
        """
        Normalize weight or factor dictionary to the employed format.
        """
        if w is None:
            # use default
            w = {t: {k: default_val for k in sum_stat_keys}}
        elif not isinstance(next(iter(w.values())), dict):
            # f is not time-dependent
            # so just create one for time t
            w = {t: w}
        return w

    @staticmethod
    def get_for_t_or_latest(w, t):
        """
        Extract values from dict for given time point.
        """
        # take last time point for which values exist
        if t not in w:
            t = max(w)
        # extract values for time point
        return w[t]


class AdaptivePNormDistance(PNormDistance):
    """
    In the p-norm distance, adapt the weights for each generation, based on
    the previous simulations. This class is motivated by [#prangle]_.

    Parameters
    ----------

    p:
        p for p-norm. Required p >= 1, p = np.inf allowed (infinity-norm).
        Default: p=2.
    initial_weights:
        Weights to be used in the initial iteration. Dictionary with
        observables as keys and weights as values.
    factors:
        As in PNormDistance.
    adaptive:
        True: Adapt distance after each iteration.
        False: Adapt distance only once at the beginning in initialize().
        This corresponds to a pre-calibration.
    scale_function:
        (data: list, x_0: float) -> scale: float. Computes the scale (i.e.
        inverse weight s = 1 / w) for a given summary statistic. Here, data
        denotes the list of simulated summary statistics, and x_0 the observed
        summary statistic. Implemented are absolute_median_deviation,
        standard_deviation (default), centered_absolute_median_deviation,
        centered_standard_deviation.
    normalize_weights:
        Whether to normalize the weights to have mean 1. This just possibly
        smoothes the decrease of epsilon and might aid numeric stability, but
        is not strictly necessary.
    max_weight_ratio:
        If not None, large weights will be bounded by the ratio times the
        smallest non-zero absolute weight. In practice usually not necessary,
        it is theoretically required to ensure convergence.
    log_file:
        A log file to store weights for each time point in. Weights are
        currently not stored in the database. The data are saved in json
        format and can be retrieved via `pyabc.storage.load_dict_from_json`.


    .. [#prangle] Prangle, Dennis. "Adapting the ABC Distance Function".
                Bayesian Analysis, 2017. doi:10.1214/16-BA1002.
    """

    def __init__(self,
                 p: float = 2,
                 initial_weights: dict = None,
                 factors: dict = None,
                 adaptive: bool = True,
                 scale_function: Callable = None,
                 normalize_weights: bool = True,
                 max_weight_ratio: float = None,
                 log_file: str = None):
        # call p-norm constructor
        super().__init__(p=p, weights=None, factors=factors)

        self.initial_weights = initial_weights
        self.factors = factors
        self.adaptive = adaptive

        if scale_function is None:
            scale_function = standard_deviation
        self.scale_function = scale_function

        self.normalize_weights = normalize_weights
        self.max_weight_ratio = max_weight_ratio
        self.log_file = log_file

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
                   get_all_sum_stats: Callable[[], List[dict]],
                   x_0: dict = None):
        """
        Initialize weights.
        """
        super().initialize(t, get_all_sum_stats, x_0)
        self.x_0 = x_0

        # initial weights pre-defined
        if self.initial_weights is not None:
            self.weights[t] = self.initial_weights
            return

        # execute function
        all_sum_stats = get_all_sum_stats()

        # update weights from samples
        self._update(t, all_sum_stats)

    def update(self,
               t: int,
               get_all_sum_stats: Callable[[], List[dict]]):
        """
        Update weights.
        """
        if not self.adaptive:
            return False

        # execute function
        all_sum_stats = get_all_sum_stats()

        self._update(t, all_sum_stats)

        return True

    def _update(self,
                t: int,
                all_sum_stats: List[dict]):
        """
        Here the real update of weights happens.
        """
        # retrieve keys
        keys = self.x_0.keys()

        # number of samples
        n_samples = len(all_sum_stats)

        # to-be-filled-and-appended weights dictionary
        w = {}

        for key in keys:
            # prepare list for key
            current_list = []
            for j in range(n_samples):
                if key in all_sum_stats[j]:
                    current_list.append(all_sum_stats[j][key])

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
        self.weights[t] = w

        # logging
        self.log(t)

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
                "factors": self.factors,
                "adaptive": self.adaptive,
                "scale_function": self.scale_function.__name__,
                "normalize_weights": self.normalize_weights,
                "max_weight_ratio": self.max_weight_ratio}

    def log(self, t: int) -> None:
        logger.debug(f"updated weights[{t}] = {self.weights[t]}")

        if self.log_file:
            save_dict_to_json(self.weights, self.log_file)


class AggregatedDistance(Distance):
    """
    Aggregates a list of distance functions, all of which may work on subparts
    of the summary statistics. Then computes and returns the weighted sum of
    the distance values generated by the various distance functions.

    All class functions are propagated to the children and the obtained
    results aggregated appropriately.
    """

    def __init__(
            self,
            distances: List[Distance],
            weights: Union[List, dict] = None,
            factors: Union[List, dict] = None):
        """
        Parameters
        ----------

        distances: List
            The distance functions to apply.
        weights: Union[List, dict], optional (default = [1,...])
            The weights to apply to the distances when taking the sum. Can be
            a list with entries in the same order as the distances, or a
            dictionary of lists, with the keys being the single time points
            (if the weights should be iteration-specific).
        factors: Union[List, dict], optional (dfault = [1,...])
            Scaling factors that the weights are multiplied with. The same
            structure applies as to weights.
            If None is passed, a factor of 1 is considered for every summary
            statistic.
            Note that in this class, factors are superfluous as everything can
            be achieved with weights alone, however in subclsses the factors
            can remain static while weights adapt over time, allowing for
            greater flexibility.
        """
        super().__init__()

        if not isinstance(distances, list):
            distances = [distances]
        self.distances = [to_distance(distance) for distance in distances]

        self.weights = weights
        self.factors = factors

    def initialize(
            self,
            t: int,
            get_all_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        super().initialize(t, get_all_sum_stats, x_0)
        for distance in self.distances:
            distance.initialize(t, get_all_sum_stats, x_0)
        self.format_weights_and_factors(t)

    def configure_sampler(
            self,
            sampler: Sampler):
        """
        Note: `configure_sampler` is applied by all distances sequentially,
        so care must be taken that they perform no contradictory operations
        on the sampler.
        """
        for distance in self.distances:
            distance.configure_sampler(sampler)

    def update(
            self,
            t: int,
            get_all_sum_stats: Callable[[], List[dict]]) -> bool:
        """
        The `sum_stats` are passed on to all distance functions, each of
        which may then update using these. If any update occurred, a value
        of True is returned indicating that e.g. the distance may need to
        be recalculated since the underlying distances changed.
        """
        return any(distance.update(t, get_all_sum_stats)
                   for distance in self.distances)

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        """
        Applies all distance functions and computes the weighted sum of all
        obtained values.
        """
        values = np.array([
            distance(x, x_0, t, par) for distance in self.distances
        ])
        self.format_weights_and_factors(t)
        weights = AggregatedDistance.get_for_t_or_latest(self.weights, t)
        factors = AggregatedDistance.get_for_t_or_latest(self.factors, t)
        return float(np.dot(weights * factors, values))

    def get_config(self) -> dict:
        """
        Return configuration of the distance.

        Returns
        -------

        config: dict
            Dictionary describing the distance.
        """
        config = {}
        for j, distance in enumerate(self.distances):
            config[f'Distance_{j}'] = distance.get_config()
        return config

    def format_weights_and_factors(self, t):
        self.weights = AggregatedDistance.format_dict(
            self.weights, t, len(self.distances))
        self.factors = AggregatedDistance.format_dict(
            self.factors, t, len(self.distances))

    @staticmethod
    def format_dict(w, t, n_distances, default_val=1.):
        """
        Normalize weight or factor dictionary to the employed format.
        """
        if w is None:
            # use default
            w = {t: default_val * np.ones(n_distances)}
        elif not isinstance(w, dict):
            # f is not time-dependent
            # so just create one for time t
            w = {t: np.array(w)}
        return w

    @staticmethod
    def get_for_t_or_latest(w, t):
        """
        Extract values from dict for given time point.
        """
        # take last time point for which values exist
        if t not in w:
            t = max(w)
        # extract values for time point
        return w[t]


class AdaptiveAggregatedDistance(AggregatedDistance):
    """
    Adapt the weights of `AggregatedDistances` automatically over time.

    Parameters
    ----------
    distances:
        As in AggregatedDistance.
    initial_weights:
        Weights to be used in the initial iteration. List with
        a weight for each distance function.
    factors:
        As in AggregatedDistance.
    adaptive:
        True: Adapt weights after each iteration.
        False: Adapt weights only once at the beginning in initialize().
        This corresponds to a pre-calibration.
    scale_function:
        Function that takes a list of floats, namely the values obtained
        by applying one of the distances passed to a set of samples,
        and returns a single float, namely the weight to apply to this
        distance function. Default: scale_span.
    log_file:
        A log file to store weights for each time point in. Weights are
        currently not stored in the database. The data are saved in json
        format and can be retrieved via `pyabc.storage.load_dict_from_json`.
    """

    def __init__(
            self,
            distances: List[Distance],
            initial_weights: List = None,
            factors: Union[List, dict] = None,
            adaptive: bool = True,
            scale_function: Callable = None,
            log_file: str = None):
        super().__init__(distances=distances)
        self.initial_weights = initial_weights
        self.factors = factors
        self.adaptive = adaptive
        self.x_0 = None
        if scale_function is None:
            scale_function = span
        self.scale_function = scale_function
        self.log_file = log_file

    def initialize(self,
                   t: int,
                   get_all_sum_stats: Callable[[], List[dict]],
                   x_0: dict = None):
        """
        Initialize weights.
        """
        super().initialize(t, get_all_sum_stats, x_0)
        self.x_0 = x_0

        if self.initial_weights is not None:
            self.weights[t] = self.initial_weights
            return

        # execute function
        all_sum_stats = get_all_sum_stats()

        # update weights from samples
        self._update(t, all_sum_stats)

    def update(self,
               t: int,
               get_all_sum_stats: Callable[[], List[dict]]):
        """
        Update weights based on all simulations.
        """
        super().update(t, get_all_sum_stats)

        if not self.adaptive:
            return False

        # execute function
        all_sum_stats = get_all_sum_stats()

        self._update(t, all_sum_stats)

        return True

    def _update(self,
                t: int,
                sum_stats: List[dict]):
        """
        Here the real update of weights happens.
        """
        # to-be-filled-and-appended weights dictionary
        w = []

        for distance in self.distances:
            # apply distance to all samples
            current_list = [
                distance(sum_stat, self.x_0)
                for sum_stat in sum_stats
            ]
            # compute scaling
            scale = self.scale_function(current_list)

            # compute weight (inverted scale)
            if np.isclose(scale, 0):
                # This means that either the summary statistic is not in the
                # samples, or that all simulations were identical. In either
                # case, it should be safe to ignore this summary statistic.
                w.append(0)
            else:
                w.append(1 / scale)

        # add to w attribute, at time t
        self.weights[t] = np.array(w)

        # logging
        self.log(t)

    def log(self, t: int) -> None:
        logger.debug(f"updated weights[{t}] = {self.weights[t]}")

        if self.log_file:
            save_dict_to_json(self.weights, self.log_file)


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
                   get_all_sum_stats: Callable[[], List[dict]],
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
        return np.asarray([x[key] for key in self.measures_to_use])

    def _calculate_whitening_transformation_matrix(self, sum_stats):
        samples_vec = np.asarray([self._dict_to_vect(x)
                                  for x in sum_stats])
        # samples_vec is an array of shape nr_samples x nr_features
        means = samples_vec.mean(axis=0)
        centered = samples_vec - means
        covariance = centered.T.dot(centered)
        w, v = la.eigh(covariance)
        self._whitening_transformation_matrix = (
            v.dot(np.diag(1. / np.sqrt(w))).dot(v.T))

    def initialize(self,
                   t: int,
                   get_all_sum_stats: Callable[[], List[dict]],
                   x_0: dict = None):
        super().initialize(t, get_all_sum_stats, x_0)

        # execute function
        all_sum_stats = get_all_sum_stats()

        self._calculate_whitening_transformation_matrix(all_sum_stats)

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
                   get_all_sum_stats: Callable[[], List[dict]],
                   x_0: dict = None):
        super().initialize(t, get_all_sum_stats, x_0)

        # execute function
        all_sum_stats = get_all_sum_stats()

        self._calculate_normalization(all_sum_stats)

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
        return np.percentile(parameter_list,
                             100 - PercentileDistance.PERCENTILE)

    @staticmethod
    def lower(parameter_list):
        return np.percentile(parameter_list,
                             PercentileDistance.PERCENTILE)

    def get_config(self):
        config = super().get_config()
        config["PERCENTILE"] = self.PERCENTILE
        return config
