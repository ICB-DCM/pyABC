import numpy as np
import pandas as pd
import logging
from typing import Callable, List, Union
from .base import Epsilon
from ..weighted_statistics import weighted_quantile


logger = logging.getLogger("Epsilon")


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
        super().__init__()
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
        super().__init__()
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

        super().__init__()
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
                   get_weighted_distances: Callable[[], pd.DataFrame],
                   get_all_records: Callable[[], List[dict]],
                   max_nr_populations: int,
                   acceptor_config: dict):
        if self._initial_epsilon != 'from_sample':
            # safety check in __call__
            return

        # execute function
        weighted_distances = get_weighted_distances()

        # initialize epsilon
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
               get_weighted_distances: Callable[[], pd.DataFrame],
               get_all_records: Callable[[], List[dict]],
               acceptance_rate: float,
               acceptor_config: dict):
        """
        Compute quantile of the (weighted) distances given in population,
        and use this to update epsilon.
        """
        # execute function
        weighted_distances = get_weighted_distances()

        # update epsilon
        self._update(t, weighted_distances)

        # logger
        logger.debug("new eps, t={}, eps={}".format(t, self._look_up[t]))

    def _update(self,
                t: int,
                weighted_distances: pd.DataFrame):
        """
        Here the real update happens, based on the weighted distances.
        """

        # extract distances
        distances = weighted_distances.distance.values

        # extract weights
        if self.weighted:
            weights = weighted_distances.w.values.astype(float)
            # The sum of the weighted distances is larger than 1 if more than
            # a single simulation per parameter is performed.
            # Re-normalize in this case.
            weights /= weights.sum()
        else:
            len_distances = len(distances)
            weights = np.ones(len_distances) / len_distances

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
