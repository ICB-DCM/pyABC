from abc import ABC, abstractmethod
from typing import Callable, TypeVar
from pyabc.population import FullInfoParticle, Population

A = TypeVar('A')


class SampleOptions:

    def __init__(self):
        self.sample_one = None # Callable[[], A]
        self.simulate_eval_one = None # Callable[[A], FullInfoParticle]
        self.n = None # int
        self.record_all_summary_statistics = False


class Sample:
    """
    A Sample is created and filled during the sampling process by the Sampler.

    Properties
    ----------

    accepted_population: Population
        Contains all accepted particles.

    all_summary_statistics_list: List[dict]
        Contains all summary statistics created during the sampling process.
    """

    def __init__(self, sample_options: SampleOptions):
        self.accepted_particles = list()
        self.all_summary_statistics_list = list()
        self.sample_options = sample_options

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
        if self.sample_options.record_all_summary_statistics:
            self.all_summary_statistics_list.extend(
                full_info_particle.all_summary_statistics_list)

    def __add__(self, other):
        sample = Sample(self.sample_options)
        sample.accepted_particles = self.accepted_particles \
            + other.accepted_particles
        sample.all_summary_statistics_list = \
            self.all_summary_statistics_list \
            + other.all_summary_statistics_list

        return sample

    def get_accepted_population(self):
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

    @abstractmethod
    def sample_until_n_accepted(self, sample_options: SampleOptions) -> Sample:
        """
        Parameters
        ----------
        sample_one: Callable[[], A]
            A function which takes no arguments and returns
            a proposal parameter :math:`\\theta`.

        simulate_eval_one: Callable[[A], ValidParticle]
            A function which takes as sole argument a proposal
            parameter :math:`\\theta` as returned by `sample_one`.
            It returns a :class:`FullInfoParticle` containing the summary
            statistics. In a field accepted, this particle returns also the
            information whether it got accepted.

        record_all_particles: bool
            True if all particles should be recorded, False if only record
            accepted particles.

        n: int
            The number of samples to be accepted, i.e. the population size.

        Returns
        -------
        sample: :class:`Sample`
            The generated sample.
        """
