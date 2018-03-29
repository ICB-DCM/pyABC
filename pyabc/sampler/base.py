from abc import ABC, abstractmethod
from pyabc.population import Particle, Population
from typing import List


class Sample:
    """
    A Sample is created and filled during the sampling process by the Sampler.

    Parameters
    ----------

    record_rejected_summary_statistics: bool
        Whether to record the summary statistics of rejected particles as well.



    Properties
    ----------

    record_rejected_sum_stat: bool
        Whether to record summary statistics of the rejected samples as well.
    """

    def __init__(self, record_rejected_sum_stat: bool):
        self._particles = []
        self.record_rejected_sum_stat = record_rejected_sum_stat

    @property
    def all_summary_statistics(self):
        """

        Returns
        -------

        List of all summary statistics, of accepted and rejected particles.
        """
        return sum((particle.all_sum_stats
                    for particle in self._particles), [])

    @property
    def _accepted_particles(self) -> List[Particle]:
        """

        Returns
        -------

        List of only the accepted particles.
        """
        return [particle.copy()
                for particle in self._particles if particle.accepted]

    def append(self, particle: Particle):
        """
        Add new particle to the sample.


        Parameters
        ----------

        particle: Particle
            Sampled particle containing all information needed later.
        """

        # add to population if accepted
        if particle.accepted or self.record_rejected_sum_stat:
            self._particles.append(particle)

    def __add__(self, other: "Sample"):
        sample = self.__class__(self.record_rejected_sum_stat)
        sample._particles = self._particles + other._particles
        return sample

    @property
    def n_accepted(self):
        return len(self._accepted_particles)

    def get_accepted_population(self) -> Population:
        """
        Returns
        -------

        A population of only the accepted particles.

        :return:
            A Population object.
        """
        return Population(self._accepted_particles)


class SampleFactory:
    def __init__(self, record_all_sum_stats):
        self.record_all_sum_stats = record_all_sum_stats

    def __call__(self):
        return Sample(self.record_all_sum_stats)

    def require_all_sum_stats(self):
        self.record_all_sum_stats = True


class Sampler(ABC):
    """
    Abstract Sampler base class.

    Produce valid particles: :class:`pyabc.parameters.ValidParticle`.

    Properties
    ----------

    nr_evaluations_: int
        This is set after a population and counts the total number
        of model evaluations. This can be used to calculate the acceptance
        rate.
    """
    def __init__(self):
        self.nr_evaluations_ = 0
        self.sample_factory = SampleFactory(False)

    def require_all_sum_stats(self):
        self.sample_factory.require_all_sum_stats()

    def _create_empty_sample(self) -> Sample:
        return self.sample_factory()

    @abstractmethod
    def sample_until_n_accepted(self, n, simulate_one) -> Sample:
        """
        Parameters
        ----------

        Returns
        -------

        sample: :class:`Sample`
            The generated sample.
        """
