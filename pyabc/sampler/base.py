from abc import ABC, abstractmethod
from pyabc.population import FullInfoParticle, Population


class Sample:
    """
    A Sample is created and filled during the sampling process by the Sampler.

    Parameters
    ----------

    record_rejected_summary_statistics: bool
        Whether to record the summary statistics of rejected particles as well.


    Properties
    ----------

    accepted_population: Population
        Contains all accepted particles.

    all_summary_statistics_list: List[dict]
        Contains all summary statistics created during the sampling process.
    """

    def __init__(self, record_rejected_sum_stat: bool):
        self.accepted_particles = []
        self._all_summary_statistics_list = []
        self.record_rejected_sum_stat = record_rejected_sum_stat

    @property
    def all_summary_statistics_list(self):
        if self.record_rejected_sum_stat:
            return self._all_summary_statistics_list
        return sum((particle.summary_statistics_list
                    for particle in self.accepted_particles), [])

    def append(self, full_info_particle: FullInfoParticle):
        """
        Add new particle to sample and handle all_summary_statistics_list.
        Checks itself based on the particle.accepted flag whether the particle
        is accepted.

        :param full_info_particle:
            Sampled particle containing all information needed later.
        """

        # add to population if accepted
        if full_info_particle.accepted:
            self.accepted_particles.append(full_info_particle.to_particle())

        # keep track of all summary statistics sampled
        if self.record_rejected_sum_stat:
            self._all_summary_statistics_list.extend(
                full_info_particle.all_summary_statistics_list)

    def __add__(self, other):
        """
        Sum function.
        :param other:
        :return:
        """
        sample = self.__class__(self.record_rejected_sum_stat)
        sample.accepted_particles = self.accepted_particles \
            + other.accepted_particles
        sample._all_summary_statistics_list = \
            self._all_summary_statistics_list \
            + other.all_summary_statistics_list

        return sample

    def get_accepted_population(self):
        """
        Create and return a population from the internal list of accepted
        particles.

        :return:
            A Population object.
        """

        return Population(self.accepted_particles)


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
        self._record_all_sum_stats = False

    def require_all_sum_stats(self):
        self._record_all_sum_stats = True

    def _create_empty_sample(self) -> Sample:
        return Sample(self._record_all_sum_stats)

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
