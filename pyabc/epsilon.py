"""
Acceptance threshold scheduling strategies
==========================================

Acceptance thresholds (= epsilon) can be calculated based on the distances from
the observed data, can follow a pre-defined list, can be constant, or can have
a user-defined implementation.
"""


import scipy as sp
import logging
import json
from abc import ABC, abstractmethod
from .storage import History
from .weighted_statistics import weighted_quantile
from pyabc.population import Population
from typing import Callable, List, Union
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
            List of dictionaries containing the summary statistics.

        distance_to_ground_truth_function: Callable[[dict], float]
            One of the distance functions pre evaluated at its second argument
            (the one representing the measured data).
            E.g. something like lambda x: distance_function(x, x_measured)

        """
        pass

    @abstractmethod
    def __call__(self, t: int,
                 history: History,
                 latest_population: Population =None) -> float:
        """
        Possibly calculate and return current epsilon / epsilon at time t.

        Parameters
        ----------
        t: int
            The population number. Counting is zero based. So the first
            population has t=0.

        history: History
            ABC history object. Can be used to query summary statistics to
            set the epsilon

        latest_population: Population
            The latest population, which may be different from the one stored
            in history.

        :return:
        eps: float
            The new epsilon for population ``t``.

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

    def __call__(self, t: int,
                 history: History,
                 latest_population: Population =None):
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

    def __call__(self, t: int,
                 history: History,
                 latest_population: Population =None):
        return self.epsilon_values[t]


class QuantileEpsilon(Epsilon):
    """
    Calculate epsilon as alpha-quantile of the distances from the last
    population.

    Parameters
    ----------

    initial_epsilon: Union[str, int]

        * If 'from_sample', then the initial quantilee is calculated from
          a sample of the current population size from the prior distribution.
        * If a number is given, this number is used.

    alpha: float
        The alpha-quantile to be used, e.g. alpha=0.5 means median.

    quantile_multiplier: float
        Multiplies the quantile by that number. also applies it
        to the initial quantile if it is calculated from samples.
        However, it does **not** apply to the initial quantile if
        it is given as a number.

    weighted: bool
        Flag indicating whether the new epsilon should be computed using
        weighted (True, default) or non-weighted (False) distances.

    This strategy works even if the posterior is multi-modal.
    Note that the acceptance threshold calculation is based on the distance
    to the observation, not on the parameters which generated data with that
    distance.
    If completely different parameter sets produce equally good samples,
    the distances of their samples to the ground truth data should be
    comparable.
    The idea behind weighting is that the probability p_k of obtaining a
    distance eps_k in the next generation should be proportional to the
    weight w_k of respective particle k in the current generation. Both
    weighted and non-weighted median should lead to correct results.
    """

    def __init__(self,
                 initial_epsilon: Union[str, int, float]='from_sample',
                 alpha: float =0.5,
                 quantile_multiplier: float =1,
                 weighted: bool =True):
        eps_logger.debug(
            "init quantile_epsilon initial_epsilon={}, quantile_multiplier={}"
            .format(initial_epsilon, quantile_multiplier))
        super().__init__()
        self.alpha = alpha
        self._initial_epsilon = initial_epsilon
        self.quantile_multiplier = quantile_multiplier

        self.weighted = weighted
        self._look_up = {}

        if self.alpha > 1 or self.alpha <= 0:
            raise ValueError("It must be 0 < alpha <= 1")

    def get_config(self):
        config = super().get_config()
        config.update({"initial_epsilon": self._initial_epsilon,
                       "quantile_multiplier": self.quantile_multiplier})

        return config

    def initialize(self,
                   sample_from_prior, distance_to_ground_truth_function):
        super().initialize(sample_from_prior,
                           distance_to_ground_truth_function)

        # calculate initial epsilon if not given
        if self._initial_epsilon == 'from_sample':
            distances = sp.asarray([distance_to_ground_truth_function(x)
                                    for x in sample_from_prior])
            eps_t0 = weighted_quantile(points=distances, alpha=self.alpha)\
                * self.quantile_multiplier
            self._look_up = {0: eps_t0}
        else:
            self._look_up = {0: self._initial_epsilon}

        # logging
        eps_logger.info("initial epsilon is {}".format(self._look_up[0]))

    def __call__(self, t: int,
                 history: History,
                 latest_population: Population =None) -> float:
        """
        Compute new epsilon from input, and return it.
        """
        try:
            eps = self._look_up[t]
        except KeyError:
            # this will be the usual case after the first iteration
            self._update(t, history, latest_population)
            eps = self._look_up[t]

        return eps

    def _update(self, t: int,
                history: History,
                latest_population: Population =None):
        """
        Compute quantile of the (weighted) distances given in population,
        and use this to update epsilon.
        """

        # If latest_population is None, e.g. after smc.load(), read latest
        # weighted distances from history, else use the passed latest
        # population.
        if latest_population is None:
            df_weighted = history.get_weighted_distances(None)
        else:
            df_weighted = latest_population.get_weighted_distances()

        distances = df_weighted.distance.as_matrix()
        len_distances = len(distances)

        if self.weighted:
            weights = df_weighted.w.as_matrix()
            # The sum of the weighted distances is larger than 1 if more than
            # a single simulation per parameter is performed.
            # Re-normalize in this case.
            weights /= weights.sum()
        else:
            weights = sp.ones(len_distances) / len_distances

        quantile = weighted_quantile(
            points=distances, weights=weights, alpha=self.alpha)

        self._look_up[t] = quantile * self.quantile_multiplier

        # logger
        eps_logger.debug("new eps, t={}, eps={}".format(t, self._look_up[t]))


class MedianEpsilon(QuantileEpsilon):
    """
    Calculate epsilon as median of the distances from the last population.
    """

    def __init__(self, initial_epsilon: Union[str, int, float]='from_sample',
                 median_multiplier: float =1,
                 weighted: bool =True):

        super().__init__(initial_epsilon=initial_epsilon,
                         alpha=0.5,
                         quantile_multiplier=median_multiplier,
                         weighted=weighted)
