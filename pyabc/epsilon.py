"""

Acceptance threshold scheduling strategies.

Acceptance thresholds can calculated based on the distances from the
observed data or can follow a pre-defined list, can be constant or can have
a user-defined implementation.
"""


import scipy as sp
import logging
import json
from abc import ABC, abstractmethod
from .storage import History
from .weighted_statistics import weighted_median
from typing import List, Callable, Union
eps_logger = logging.getLogger("Epsilon")


class Epsilon(ABC):
    """
    Abstract epsilon base class.

    This class encapsulates a strategy for setting a new epsilon for
    each new population.
    """
    def initialize(self, sample_from_prior: List[dict],
                   distance_to_ground_truth_function: Callable[[dict], float]):
        """
        This method is called by the ABCSMC framework before the first usage
        of the epsilon
        and can be used to calibrate it to the statistics of the samples.

        The default implementation is to do nothing. It is not necessary
        to implement this method.

        Parameters
        ----------

        sample_from_prior: List[dict]
            List of dictionaries containng the summary statistics.

        distance_to_ground_truth_function: Callable[[dict], float]
            One of the distance functions pre evaluated at its second argument
            (the one representing the measured data).
            E.g. something like lambda x: distance_funciton(x, x_measured)

        """

    def get_config(self):
        """
        Return configuration of the distance function.

        Returns
        -------

        config: dict
            Dictionary describing the distance function.
        """
        return {"name": self.__class__.__name__}

    def to_json(self):
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

    @abstractmethod
    def __call__(self, t: int, history: History):
        """

        Parameters
        ----------
        t: int
            The population number. Counting is zero based. So the first
            population has t=0.

        history: History
            ABC history object. Can be used to query summary statistics to
            set the epsilon

        Returns
        -------

        eps: float
            The new epsilon for population ``t``.
        """


class ConstantEpsilon(Epsilon):
    """
    Keep epsilon constant over all populations.
    This acceptance threshold scheduling strategy is most likely only
    interesting for debugging purposes.

    Parameters
    ----------

    constant_epsilon_value: float
        The epsilon value for all populations
    """
    def __init__(self, constant_epsilon_value: float):
        super().__init__()
        self.constant_epsilon_value = constant_epsilon_value

    def get_config(self):
        config = super().get_config()
        config["constant_epsilon_value"] = self.constant_epsilon_value
        return config

    def __call__(self, t, history):
        return self.constant_epsilon_value


class ListEpsilon(Epsilon):
    """
    Return epsilon values from a predefined list

    Parameters
    ----------

    values: List[float]
        List of epsilon values.
        ``values[t]`` is the value for population t.
    """
    def __init__(self, values: List[float]):
        super().__init__()
        self.epsilon_values = list(values)

    def get_config(self):
        config = super().get_config()
        config["epsilon_values"] = self.epsilon_values
        return config

    def __call__(self, t, history):
        return self.epsilon_values[t]


class MedianEpsilon(Epsilon):
    """
    Calculate epsilon as median of the distances from the last population.

    Parameters
    ----------

    initial_epsilon: Union[str, int]

        * If 'from_sample', then the initial median is calculated from
          a sample of the current population size from the priro distribution.
        * If a number is given, this number is used.

    median_multiplier: float
        Multiplies the median by that number. also applies it
        to the initial median if it is calculated from samples.
        However, it does **not** apply to the initial median if
        it is given as a number.


    This strategy works even if the posterior is multi-modal.
    Note that the acceptance threshold calculation is based on the distance
    to the observation, not on the parameters which generated data with that
    distance.
    If completely different parameter sets produce equally good samples
    the distances of their samples to the ground truth data should be
    comparable.
    """

    def __init__(self, initial_epsilon: Union[str, int, float]='from_sample',
                 median_multiplier: float =1):
        eps_logger.debug(
            "init medianepsilon initial_epsilon={}, median_multiplier={}"
            .format(initial_epsilon, median_multiplier))
        super().__init__()
        self._initial_epsilon = initial_epsilon
        self.median_multiplier = median_multiplier
        self._look_up = {}

    def get_config(self):
        config = super().get_config()
        config.update({"initial_epsilon": self._initial_epsilon,
                       "median_multiplier": self.median_multiplier})
        return config

    def initialize(self, sample_from_prior, distance_to_ground_truth_function):
        super().initialize(sample_from_prior,
                           distance_to_ground_truth_function)
        eps_logger.debug("calc initial epsilon")
        # calculate initial epsilon if not given
        if self._initial_epsilon == 'from_sample':
            distances = sp.asarray([distance_to_ground_truth_function(x)
                                    for x in sample_from_prior])
            eps_t0 = sp.median(distances) * self.median_multiplier
            self._look_up = {0: eps_t0}
        else:
            self._look_up = {0: self._initial_epsilon}

        eps_logger.info("initial epsilon is {}".format(self._look_up[0]))

    def __call__(self, t, history):
        try:
            return self._look_up[t]
        except KeyError:
            df_weighted = history.get_weighted_distances(None)
            median = weighted_median(
                df_weighted.distance.as_matrix(), df_weighted.w.as_matrix())
            self._look_up[t] = median * self.median_multiplier
            eps_logger.debug("new eps, t={}, eps={}"
                             .format(t, self._look_up[t]))
            return self._look_up[t]
