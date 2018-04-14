"""
Distance functions
==================

Distance functions measure closeness of observed and sampled data.
For custom distance functions, either pass a plain function to ABCSMC or
subclass the DistanceFunction class if finer grained configuration is required.
"""

import json
import scipy as sp
from scipy import linalg as la
from abc import ABC, abstractmethod
from typing import List
import math
import statistics
import logging
from .sampler import Sampler
df_logger = logging.getLogger("DistanceFunction")


class DistanceFunction(ABC):
    """
    Abstract base class for distance functions.

    Any other distance function should inherit from this class.
    """

    def __init__(self):
        """
        Default constructor.
        """

    def initialize(self, sample_from_prior: List[dict]):
        """
        This method is called by the ABCSMC framework before the first
        usage of the distance function
        and can be used to calibrate it to the statistics of the samples.

        The default implementation is to do nothing.

        This method is not called again when an ABC run is resumed via
        ABCSCM.run(), so the user has to make sure that custom distance
        functions are ready for calling the other methods.

        Parameters
        ----------

        sample_from_prior: List[dict]
            List of dictionaries containing the summary statistics.
        """

    def configure_sampler(self, sampler: Sampler):
        """
        This is called by the ABCSMC class and gives the distance function
        the opportunity to configure the sampler.
        For example, the distance function might request the sampler
        to also return rejected particles and their summary statistics
        in order to adapt the distance functions to the statistics
        of the sample.

        The default is to do nothing.

        Parameters
        ----------
        Sampler: Sampler
            The Sampler.
        """

    def update(self, simulations_all: List[dict]) -> bool:
        """
        Update the distance function. Default: Do nothing.

        :param simulations_all:
            List of all simulations (summary statistics).

        :return:
            True: If distance function has changed.
            False: If distance function has not changed (default).
        """
        return False

    @abstractmethod
    def __call__(self, x: dict, x_0: dict) -> float:
        """
        Abstract method. This method has to be overwritten by
        all concrete implementations.

        Evaluate the distance of the tentatively samples particle to the
        measured data.

        Parameters
        ----------

        x: dict
            Summary statistics of the tentatively sampled parameter.

        x_0: dict
            Summary statistics of the measured data.

        Returns
        -------

        distance: float
            Attributes distance of the tentatively sampled particle
            from the measured data.
        """

    def get_config(self) -> dict:
        """
        Return configuration of the distance function.

        Returns
        -------

        config: dict
            Dictionary describing the distance function.
        """

        return {"name": self.__class__.__name__}

    def to_json(self) -> str:
        """
        Return JSON encoded configuration of the distance function.

        Returns
        -------

        json_str: str
            JSON encoded string describing the distance function.
            The default implementation is to try to convert the dictionary
            returned my ``get_config``.
        """

        return json.dumps(self.get_config())


class NoDistance(DistanceFunction):
    """
    Implements a kind of null object as distance function.
    """

    def __call__(self, x: dict, x_0: dict) -> float:
        raise Exception("{} is not intended to be called."
                        .format(self.__class__.__name__))


class SimpleFunctionDistance(DistanceFunction):
    """
    This is a wrapper around a simple function which calculates the distance.
    If a function is passed to the ABCSMC class, then it is converted to
    an instance of the SimpleFunctionDistance class.
    """

    def __init__(self, function):
        super().__init__()
        self.function = function

    def __call__(self, x, y):
        return self.function(x, y)

    def get_config(self):
        conf = super().get_config()
        try:
            conf["name"] = self.function.__name__
        except AttributeError:
            try:
                conf["name"] = self.function.__class_.__name__
            except AttributeError:
                pass
        return conf


def to_distance(maybe_distance_function):
    """

    Parameters
    ----------
    maybe_distance_function: either a callable, which takes two arguments or
    a DistanceFunction instance

    Returns
    -------

    """
    if maybe_distance_function is None:
        return NoDistance()

    if isinstance(maybe_distance_function, DistanceFunction):
        return maybe_distance_function
    return SimpleFunctionDistance(maybe_distance_function)


class PNormDistance(DistanceFunction):
    """
    Use weighted p-norm

    .. math::

        d(x, y) =\
         \\left[\\sum_{i} \\left w_i| x_i-y_i \\right|^{p} \\right]^{1/p}

    to compute distances between sets of summary statistics.

    Parameters
    ----------

    p: float
        p for p-norm. Required p >= 1, p = math.inf allowed (infinity-norm).

    w: dict
        Numeric weights associated with summary statistics. If none is
        passed, a weight of 1 is considered for every summary statistic.
    """

    def __init__(self, p: float, w: dict=None):
        super().__init__()
        if p < 1:
            raise ValueError("It must be p >= 1")
        self.p = p
        self.w = w

    def __call__(self, x: dict, y: dict):
        # make sure weights are initialized
        if self.w is None:
            self._initialize_weights(x.keys())

        # compute p-norm distance
        if self.p == math.inf:
            return max(abs(self.w[key]*(x[key]-y[key]))
                       for key in self.w.keys())
        else:
            return pow(
                sum(pow(abs(self.w[key]*(x[key]-y[key])), self.p)
                    for key in self.w.keys()),
                1/self.p)

    def _initialize_weights(self, summary_statistics_keys):
        """
        Init weights to 1 for every summary statistic.
        """
        self.w = {k: 1 for k in summary_statistics_keys}


class EuclideanDistance(PNormDistance):
    """
    Comfort class to use Euclidean norm as p-norm in PNormDistance.
    """

    def __init__(self):
        super().__init__(2)


class AdaptivePNormDistance(PNormDistance):
    """
    Use a weighted p-norm to compute distances between sets of summary
    statistics.

    Parameters
    ----------

    p: float
        p for p-norm. Required p >= 1, p = math.inf allowed (infinity-norm).

    adaptive: bool
        True: Adapt distance after each iteration.
        False: Adapt distance only once at the beginning in initialize().

    scale_type: int
        What measure to use for deviation. Values as in SCALE_... constants.
    """

    # mean absolute deviation
    SCALE_TYPE_MAD = 0

    # standard deviation
    SCALE_TYPE_SD = 1

    def __init__(self,
                 p: float,
                 adaptive: bool=True,
                 scale_type: int=SCALE_TYPE_MAD):
        # call p-norm constructor
        super().__init__(p)

        self.adaptive = adaptive
        self.scale_type = scale_type

    def configure_sampler(self, sampler: Sampler):
        """
        Make the sampler return also rejected summary statistics if required,
        because these are needed to get a better estimate of the summary
        statistic variabilities.

        Parameters
        ----------

        sampler: Sampler
            The sampler employed.
        """

        super().configure_sampler(sampler)
        if self.adaptive:
            sampler.sample_factory.record_all_sum_stats = True

    def initialize(self, sample_from_prior: List[dict]):
        """
        Initialize weights.

        Parameters
        ----------

        sample_from_prior: List[dict]
            A sample from the prior distribution.
        """

        super().initialize(sample_from_prior)
        # update weights from samples
        self._update(sample_from_prior)

    def update(self, all_summary_statistics_list: List[dict]):
        """
        Update weights based on all simulations. Usually called in
        each iteration.

        :param all_summary_statistics_list: List[dict]
            List of all summary statistics (also those rejected).
        """

        if not self.adaptive:
            return False

        self._update(all_summary_statistics_list)

        return True

    def _update(self, all_summary_statistics_list: List[dict]):
        """
        Here the real update of weights happens.

        :param all_summary_statistics_list: List[dict]
            List of all summary statistics (also those rejected).
        :return:
        """

        # make sure weights are initialized
        if self.w is None:
            self._initialize_weights(all_summary_statistics_list[0].keys())

        n = len(all_summary_statistics_list)

        for key in self.w.keys():
            # prepare list for key
            current_list = []
            for j in range(n):
                current_list.append(all_summary_statistics_list[j][key])

            # compute weighting
            if self.scale_type == AdaptivePNormDistance.SCALE_TYPE_MAD:
                val = median_absolute_deviation(current_list)
            elif self.scale_type == AdaptivePNormDistance.SCALE_TYPE_SD:
                val = standard_deviation(current_list)
            else:
                raise Exception(
                    "pyabc:distance_function: scale_type not recognized.")

            if val == 0:
                # in practise, this case should be rare (if only for numeric
                # reasons, so setting the weight to 1 should be safe)
                self.w[key] = 1
            else:
                self.w[key] = 1 / val

        # normalize weights to have mean 1. This has just the effect that the
        # epsilon will decrease more smoothly, but is not important otherwise.
        mean_weight = statistics.mean(list(self.w.values()))
        for key in self.w.keys():
            self.w[key] /= mean_weight

        # logging
        df_logger.debug("update distance weights = {}".format(self.w))


class AdaptiveEuclideanDistance(AdaptivePNormDistance):
    """
    Comfort class to use Euclidean norm as p-norm in WeightedPNormDistance.
    """

    def __init__(self,
                 adaptive: bool = True,
                 scale_type: int = AdaptivePNormDistance.SCALE_TYPE_MAD):
        super().__init__(2, adaptive, scale_type)


def median_absolute_deviation(data: List):
    """
    Calculate the sample `median absolute deviation (MAD)
    <https://en.wikipedia.org/wiki/Median_absolute_deviation/>`_, defined as
    median(abs(data - median(data)).

    Parameters
    ----------

    data: List
        List of data points.

    Returns
    -------

    mad
        The median absolute deviation of the data.

    """

    data_median = statistics.median(data)
    normalized_data = []
    for item in data:
        normalized_data.append(abs(item - data_median))
    mad = statistics.median(normalized_data)
    return mad


def standard_deviation(data: List):
    """
    Calculate the sample `standard deviation (SD)
    <https://en.wikipedia.org/wiki/Standard_deviation/>`_.

    Parameters
    ----------

    data: List
        List of data points.

    Returns
    -------

    sd
        The standard deviation of the data points.
    """

    sd = statistics.stdev(data)
    return sd


class DistanceFunctionWithMeasureList(DistanceFunction):
    """
    Base class for distance functions with measure list.
    This class is not functional on its own.

    Parameters
    ----------

    measures_to_use: Union[str, List[str]].
        * If set to "all", all measures are used. This is the default
        * If a list is provided, the measures in the list are used.
        * measures refers to the summary statistics.
    """

    def __init__(self, measures_to_use='all'):
        super().__init__()
        self._measures_to_use_passed_to_init = measures_to_use
        #: The measures (summary statistics) to use for distance calculation.
        self.measures_to_use = None

    def initialize(self, sample_from_prior):
        super().initialize(sample_from_prior)
        if self._measures_to_use_passed_to_init == 'all':
            self.measures_to_use = sample_from_prior[0].keys()
            raise Exception(
                "distance function from all measures not implemented.")
        else:
            self.measures_to_use = self._measures_to_use_passed_to_init

    def get_config(self):
        config = super().get_config()
        config["measures_to_use"] = self.measures_to_use
        return config


class ZScoreDistanceFunction(DistanceFunctionWithMeasureList):
    """
    Calculate distance as sum of ZScore over the selected measures.
    The measured Data is the reference for the ZScore.

    Hence

    .. math::

        d(x, y) =\
         \\sum_{i \\in \\text{measures}} \\left| \\frac{x_i-y_i}{y_i} \\right|
    """
    def __call__(self, x, y):
        return sum(abs((x[key]-y[key])/y[key]) if y[key] != 0 else
                   (0 if x[key] == 0 else sp.inf)
                   for key in self.measures_to_use) / len(self.measures_to_use)


class PCADistanceFunction(DistanceFunctionWithMeasureList):
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

    def _dict_to_to_vect(self, x):
        return sp.asarray([x[key] for key in self.measures_to_use])

    def _calculate_whitening_transformation_matrix(self, sample_from_prior):
        samples_vec = sp.asarray([self._dict_to_to_vect(x)
                                  for x in sample_from_prior])
        # samples_vec is an array of shape nr_samples x nr_features
        means = samples_vec.mean(axis=0)
        centered = samples_vec - means
        covariance = centered.T.dot(centered)
        w, v = la.eigh(covariance)
        self._whitening_transformation_matrix = (
            v.dot(sp.diag(1. / sp.sqrt(w))).dot(v.T))

    def initialize(self, sample_from_prior):
        super().initialize(sample_from_prior)
        self._calculate_whitening_transformation_matrix(sample_from_prior)

    def __call__(self, x, y):
        x_vec, y_vec = self._dict_to_to_vect(x), self._dict_to_to_vect(y)
        distance = la.norm(
            self._whitening_transformation_matrix.dot(x_vec - y_vec), 2)
        return distance


class RangeEstimatorDistanceFunction(DistanceFunctionWithMeasureList):
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

    def _calculate_normalization(self, sample_from_prior):
        measures = {name: [] for name in self.measures_to_use}
        for sample in sample_from_prior:
            for measure in self.measures_to_use:
                measures[measure].append(sample[measure])
        self.normalization = {measure:
                              self.upper(measures[measure])
                              - self.lower(measures[measure])
                              for measure in self.measures_to_use}

    def initialize(self, sample_from_prior):
        super().initialize(sample_from_prior)
        self._calculate_normalization(sample_from_prior)

    def __call__(self, x, y):
        distance = sum(abs((x[key]-y[key])/self.normalization[key])
                       for key in self.measures_to_use)
        return distance


class MinMaxDistanceFunction(RangeEstimatorDistanceFunction):
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


class PercentileDistanceFunction(RangeEstimatorDistanceFunction):
    """
    Calculate normalization 20% and 80% from percentiles as lower
    and upper margins
    """

    PERCENTILE = 20  #: The percentiles

    @staticmethod
    def upper(parameter_list):
        return sp.percentile(parameter_list,
                             100 - PercentileDistanceFunction.PERCENTILE)

    @staticmethod
    def lower(parameter_list):
        return sp.percentile(parameter_list,
                             PercentileDistanceFunction.PERCENTILE)

    def get_config(self):
        config = super().get_config()
        config["PERCENTILE"] = self.PERCENTILE
        return config


class AcceptAllDistance(DistanceFunction):
    """
    Just a mock distance function which always returns -1.
    So any sample should be accepted for any sane epsilon object.

    Can be used for testing.
    """
    def __call__(self, x, y):
        """
        Parameters
        ----------
        x: dictionary
            sample point
        y: dictionary
            measured point
        """
        return -1


class IdentityFakeDistance(DistanceFunction):
    """
    A fake distance function, which just passes
    the summary statistics on. This class assumes, that
    the model already returns the distance. This can be useful
    in cases where simulating can be stopped early when
    during the simulation some condition is reached which
    makes it impossible to accept the particle.
    """
    def __call__(self, x, y):
        return x
