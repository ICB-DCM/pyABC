from abc import ABC, abtractmethod
from typing import List, Callable
import json


class Comparator(ABC):
    """
    Abstract base class for comparator objects.

    Any comparator that computes the similarity between observed and
    simulated data should inherit from this class.
    """

    def __init__(self):
        """
        Default constructor.
        """
        pass

    def initialize(
            self,
            t: int,
            get_sum_stats: Callable[[], List[dict]],
            x_0: dict):
        """
        This method is called by the ABCSMC framework before the first
        use of the comparator (at the beginning of ABCSMC.run()),
        and can be used to calibrate it to the statistics of the samples.

        The default implementation is to do nothing.
        
        Parameters
        ----------

        t: int
            Time point for which to initilaize the comparator.

        get_sum_stats: Callable[[], List[dict]]
            Returns on command the initial summary statistics.

        x_0: dict
            The observed summary statistics.
        """

    def configure_sample(
            self,
            sampler: Sampler):
        """
        This is called by the ABCSMC class and gives the comparator
        the opportunity to configure the sampler.
        For example, the comparator might request the sampler to
        also return rejected particles in order to adapt the
        comparator to the statistics of the sample.
        
        The default is to do nothing.

        Parameters
        ----------

        sampler: Sampler
            The sampler used in ABCSMC.
        """

    def update(
            self,
            t: int,
            sum_stats: List[dict]) -> bool:
        """
        Update the comparator for the upcoming generation t.

        The default is to do nothing.

        Parameters
        ----------

        t: int
            Time point for which to update / create the comparator measure.

        sum_stats: List[dict]
            List of all summary statistics from the finished generation
            that should be used to update the comparator.

        Returns
        -------

        is_updated: bool
            Whether the comparator has changed compared to beforehand.
            Default: False.
        """
        return False

    @abstractmethod
    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int,
            par: dict) -> float:
        """
        Evaluate at time point t the distance of the summary statistics of
        the data simulated for the tentatively sampled particle to those of
        the observed data.

        Abstract method. This method has to be overwritten by
        all concrete implementations.

        Parameters
        ----------

        t: int
            Time point at which to evaluate the comparator.

        x: dict
            Summary statistics of the tentatively sampled parameter.

        x_0: dict
            Summary statistics of the observed data.

        Returns
        -------

        comparison: float
            Quantifies the comparison between the summary statistics of the
            data simulated for the tentatively sampled particle and of the
            observed data.
        """

    def get_config(self) -> dict:
        """
        Return configuration of the comparator.

        Returns
        -------

        config: dict
            Dictionary describing the comparator.
        """
        return {"name": self.__class__.__name__}

    def to_json(self) -> str:
        """
        Return JSON encoded configuration of the comparator.

        Returns
        -------

        json_str: str:
            JSON encoded string describing the comparator.
            The default implementation is to try to convert the dictionary
            returned by ``get_config``.
        """
        return json.dumps(self.get_config())
