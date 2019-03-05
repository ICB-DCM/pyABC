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
from .weighted_statistics import weighted_quantile
from typing import List, Union
import pandas

logger = logging.getLogger("Epsilon")


class Epsilon(ABC):
    """
    Abstract epsilon base class.

    This class encapsulates a strategy for setting a new epsilon for
    each new population.
    """

    def __init__(self,
                 require_initialize: bool = True):
        """
        Constructor.

        Parameters
        ----------

        require_initialize: bool, optional
            Whether the initialize() method should be called.

        """
        self.require_initialize = require_initialize

    def initialize(self,
                   t: int,
                   weighted_distances: pandas.DataFrame):
        """
        This method is called by the ABCSMC framework before the first usage
        of the epsilon and can be used to calibrate it to the statistics of the
        samples.

        This method is only called if require_initialize == True.

        Default: Do nothing.

        Parameters
        ----------

        t: int
            The time point to initialize the epsilon for.

        weighted_distances: pandas.DataFrame
            The distances for initializing the epsilon, as
            returned by Population.get_weighted_distances().
        """
        pass

    def update(self,
               t: int,
               weighted_distances: pandas.DataFrame):
        """
        Update epsilon value to be used as acceptance criterion for
        generation t.

        Default: Do nothing.

        Parameters
        ----------

        t: int
            The generation index to update / set epsilon for. Counting is
            zero-based. So the first population has t=0.

        weighted_distances: pandas.DataFrame
            The distances that should be used to update epsilon, as returned
            by Population.get_weighted_distances(). These are usually the
            distances of samples accepted in population t-1. The distances may
            differ from those used for acceptance in population t-1, if the
            distance function for population t has been updated.
        """
        pass

    @abstractmethod
    def __call__(self,
                 t: int) -> float:
        """
        Get epsilon value for generation t.

        Parameters
        ----------

        t: int
            The time point to get the epsilon threshold for.

        Returns
        -------

        eps: float
            The epsilon for population t.
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

    def __init__(self,
                 constant_epsilon_value: float):
        super().__init__(require_initialize=False)
        self.constant_epsilon_value = constant_epsilon_value

    def get_config(self):
        config = super().get_config()
        config["constant_epsilon_value"] = self.constant_epsilon_value
        return config

    def __call__(self,
                 t: int):
        return self.constant_epsilon_value


class ListEpsilon(Epsilon):
    """
    Return epsilon values from a predefined list. For every time point
    enquired later, an epsilon value must exist in the list.

    Parameters
    ----------

    values: List[float]
        List of epsilon values.
        ``values[t]`` is the value for population t.
    """

    def __init__(self,
                 values: List[float]):
        super().__init__(require_initialize=False)
        self.epsilon_values = list(values)

    def get_config(self):
        config = super().get_config()
        config["epsilon_values"] = self.epsilon_values
        return config

    def __call__(self,
                 t: int):
        return self.epsilon_values[t]


class QuantileEpsilon(Epsilon):
    """
    Calculate epsilon as alpha-quantile of the distances from the last
    population.

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

    Parameters
    ----------

    initial_epsilon: Union[str, int]
        * If 'from_sample', then the initial quantile is calculated from
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
    """

    def __init__(self,
                 initial_epsilon: Union[str, int, float] = 'from_sample',
                 alpha: float = 0.5,
                 quantile_multiplier: float = 1,
                 weighted: bool = True):

        logger.debug(
            "init quantile_epsilon initial_epsilon={}, quantile_multiplier={}"
            .format(initial_epsilon, quantile_multiplier))
        require_initialize = initial_epsilon == 'from_sample'
        super().__init__(require_initialize=require_initialize)
        self._initial_epsilon = initial_epsilon
        self.alpha = alpha
        self.quantile_multiplier = quantile_multiplier
        self.weighted = weighted
        self._look_up = {}

        if self.alpha > 1 or self.alpha <= 0:
            raise ValueError("It must be 0 < alpha <= 1")

    def get_config(self):
        config = super().get_config()
        config.update({"initial_epsilon": self._initial_epsilon,
                       "alpha": self.alpha,
                       "quantile_multiplier": self.quantile_multiplier,
                       "weighted": self.weighted})

        return config

    def initialize(self,
                   t: int,
                   weighted_distances: pandas.DataFrame):
        # called only if require_initialize == True, i.e. if not 'from_sample'

        self._update(t, weighted_distances)

        # logging
        logger.info("initial epsilon is {}".format(self._look_up[t]))

    def __call__(self,
                 t: int) -> float:
        """
        Epsilon value for time t, set before via update() method.

        Returns
        -------

        eps: float
            The epsilon value for time t (throws error if not existent).
        """

        # initialize if necessary
        if not self._look_up:
            self._set_initial_value(t)

        try:
            eps = self._look_up[t]
        except KeyError as e:
            raise KeyError("The epsilon value for time {} does not exist: {} "
                           .format(t, repr(e)))

        return eps

    def _set_initial_value(self, t: int):
        self._look_up = {t: self._initial_epsilon}

    def update(self,
               t: int,
               weighted_distances: pandas.DataFrame):
        """
        Compute quantile of the (weighted) distances given in population,
        and use this to update epsilon.
        """

        self._update(t, weighted_distances)

        # logger
        logger.debug("new eps, t={}, eps={}".format(t, self._look_up[t]))

    def _update(self,
                t: int,
                weighted_distances: pandas.DataFrame):
        """
        Here the real update happens, based on the weighted distances.
        """

        # extract distances
        distances = weighted_distances.distance.values

        # extract weights
        if self.weighted:
            weights = weighted_distances.w.values
            # The sum of the weighted distances is larger than 1 if more than
            # a single simulation per parameter is performed.
            # Re-normalize in this case.
            weights /= weights.sum()
        else:
            len_distances = len(distances)
            weights = sp.ones(len_distances) / len_distances

        # compute weighted quantile
        quantile = weighted_quantile(
            points=distances, weights=weights, alpha=self.alpha)

        # append to look_up property
        self._look_up[t] = quantile * self.quantile_multiplier


class MedianEpsilon(QuantileEpsilon):
    """
    Calculate epsilon as median of the distances from the last population.
    """

    def __init__(self,
                 initial_epsilon: Union[str, int, float] = 'from_sample',
                 median_multiplier: float = 1,
                 weighted: bool = True):
        super().__init__(initial_epsilon=initial_epsilon,
                         alpha=0.5,
                         quantile_multiplier=median_multiplier,
                         weighted=weighted)
