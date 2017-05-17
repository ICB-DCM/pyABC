"""
Distance functions
==================

Distance functions which measure closeness of observed and sampled data.
For custom distance functions, either pass a plain function to ABCSMC or
subclass the DistanceFunction class if finer grained configuration is required.
"""

import json
import scipy as sp
from scipy import linalg as la
from abc import ABC, abstractmethod
from typing import List


class DistanceFunction(ABC):
    """
    Abstract case class for distance functions.

    Any other distance function should inherit from this class.
    """
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

    def initialize(self, sample_from_prior: List[dict]):
        """
        This method is called by the ABCSMC framework before the first
        usage of the distance function
        and can be used to calibrate it to the statistics of the samples.

        The default implementation is to do nothing.

        Parameters
        ----------

        sample_from_prior: List[dict]
            List of dictionaries containng the summary statistics.
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
    Implents a kind of null object as distance function.
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
        self.function = function

    def __call__(self, x, y):
        return self.function(x, y)

    def get_config(self):
        conf = super().get_config()
        conf["name"] = self.function.__name__
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
        self._measures_to_use_passed_to_init = measures_to_use
        #: The measures (summary statistics) to use for distance calculation.
        self.measures_to_use = None

    def initialize(self, sample_from_prior):
        super().initialize(sample_from_prior)
        if self._measures_to_use_passed_to_init == 'all':
            self.measures_to_use == sample_from_prior[0].keys()
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
    in cases where simulatin can be stopped early when
    during the simulation some condition is reached which
    makes it impossible to accept the particle.
    """
    def __call__(self, x, y):
        return x
