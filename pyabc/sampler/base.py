from abc import ABC, abstractmethod
from pyabc.population import Particle, Population
from typing import List


class Sample:
    """
    A Sample is created and filled during the sampling process by the Sampler.

    Parameters
    ----------

    record_all_sum_stats: bool
        True: Record summary statistics of the rejected particles as well.
        False: Only record accepted particles.
    """

    def __init__(self, record_all_sum_stats: bool = False):
        self._particles = []
        self.record_all_sum_stats = record_all_sum_stats

    @property
    def all_sum_stats(self):
        """
        Get all summary statistics.

        Returns
        -------

        all_sum_stats: List
            Concatenation of all the all_sum_stats lists of all
            particles added and accepted to this sample via append().
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
        if particle.accepted or self.record_all_sum_stats:
            self._particles.append(particle)

    def __add__(self, other: "Sample"):
        sample = self.__class__(self.record_all_sum_stats)
        sample._particles = self._particles + other._particles
        return sample

    @property
    def n_accepted(self) -> int:
        """
        Returns
        -------

        n_accepted: int
            Number of accepted particles.
        """
        return len(self._accepted_particles)

    def get_accepted_population(self) -> Population:
        """
        Returns
        -------

        population: Population
            A population of only the accepted particles.
        """
        return Population(self._accepted_particles)


class SampleFactory:
    """
    The SampleFactory class serves as a factory class to create empty samples
    based on the parameters stored in the SampleFactory object.

    This is the class that components (like the distance function and
    epsilon) should refer to when they want to influence the sampling process.

    Parameters
    ----------

    record_all_sum_stats: bool
        Corresponds to Sample.record_all_sum_stats.
    """

    def __init__(self, record_all_sum_stats: bool = False):
        self.record_all_sum_stats = record_all_sum_stats

    def __call__(self):
        """
        Create a new empty sample.
        """
        return Sample(self.record_all_sum_stats)


class Sampler(ABC):
    """
    Abstract Sampler base class.

    Produce valid particles: :class:`pyabc.parameters.ValidParticle`.

    Parameters
    ----------

    nr_evaluations_: int
        This is set after a population and counts the total number
        of model evaluations. This can be used to calculate the acceptance
        rate.

    sample_factory: SampleFactory
        A factory to create empty samples.
    """

    def __init__(self):
        self.nr_evaluations_ = 0
        self.sample_factory = SampleFactory(
            record_all_sum_stats=False)

    def _create_empty_sample(self) -> Sample:
        return self.sample_factory()

    @abstractmethod
    def sample_until_n_accepted(self, n, simulate_one) -> Sample:
        """
        Performs the sampling, i.e. creation of a new generation (i.e.
        population) of particles.

        Parameters
        ----------

        n: int
            The number of samples to be accepted. I.e. the population size.

        simulate_one: Callable[[A], Particle]
            A function which internally performs the whole process of
            sampling parameters, simulating data, and comparing to observed
            data to check for acceptance, as indicated via the
            particle.accepted flag.

        Returns
        -------

        sample: :class:`pyabc.sampler.Sample`
            The generated sample, which contains the new population.
        """
