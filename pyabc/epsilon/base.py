import numpy as np
import pandas as pd
import json
from abc import ABC, abstractmethod
from typing import Callable, List

from ..sampler import Sampler


class Epsilon(ABC):
    """
    Abstract epsilon base class.

    This class encapsulates a strategy for setting a new epsilon for
    each new population.
    """

    def __init__(self):
        """
        Constructor.
        """
        pass

    def initialize(self,
                   t: int,
                   get_weighted_distances: Callable[[], pd.DataFrame],
                   get_all_records: Callable[[], List[dict]],
                   max_nr_populations: int,
                   acceptor_config: dict):
        """
        This method is called by the ABCSMC framework before the first usage
        of the epsilon and can be used to calibrate it to the statistics of the
        samples.

        Default: Do nothing.

        Parameters
        ----------

        t: int
            The time point to initialize the epsilon for.
        get_weighted_distances: Callable[[], pd.DataFrame]
            Returns on demand the distances for initializing the epsilon.
        get_all_records: Callable[[], List[dict]]
            Returns on demand a list of information obtained from all
            particles sampled in the previous iteration.
        max_nr_populations: int
            The maximum number of populations.
        acceptor_config: dict
            An object provided by the Acceptor class.
        """
        pass

    def configure_sampler(self, sampler: Sampler):
        """
        This is called by the ABCSMC class and gives the epsilon
        the opportunity to configure the sampler.
        For example, it might request the sampler to
        also return rejected particles in order to adapt the
        epsilon to the statistics of the sample.
        The method is called by the ABCSMC framework before the first
        use of the epsilon (at the beginning of ABCSMC.run()), after
        initialize().

        The default is to do nothing.

        Parameters
        ----------

        sampler: Sampler
            The sampler used in ABCSMC.
        """

    def update(self,
               t: int,
               get_weighted_distances: Callable[[], pd.DataFrame],
               get_all_records: Callable[[], List[dict]],
               acceptance_rate: float,
               acceptor_config: dict):
        """
        Update epsilon value to be used as acceptance criterion for
        generation t.

        Default: Do nothing.

        Parameters
        ----------

        t: int
            The generation index to update / set epsilon for. Counting is
            zero-based. So the first population has t=0.
        get_weighted_distances: Callable[[], pd.DataFrame]
            The distances that should be used to update epsilon, as returned
            by Population.get_weighted_distances(). These are usually the
            distances of samples accepted in population t-1. The distances may
            differ from those used for acceptance in population t-1, if the
            distance function for population t has been updated.
        get_all_records: Callable[[], List[dict]]
            Returns on demand a list of information obtained from all
            particles.
        acceptance_rate: float
            The current generation's acceptance rate.
        acceptor_config: dict
            An object provided by the Acceptor class.
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


class NoEpsilon(Epsilon):
    """
    Implements a kind of null object as epsilon.

    This can be used as a dummy epsilon when the Acceptor integrates the
    acceptance threshold.
    """

    def __init__(self):
        super().__init__()

    def __call__(self,
                 t: int) -> float:
        return np.nan
