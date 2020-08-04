from abc import ABC, ABCMeta, abstractmethod
import numpy as np
from typing import List, Callable

from pyabc.population import Particle, Population


class Sample:
    """
    A Sample is created and filled during the sampling process by the Sampler.

    Parameters
    ----------

    record_rejected: bool
        Whether to record rejected particles as well, along with accepted
        ones.
    ok: bool
        Whether the sampling process succeeded (usually in generating the
        requested number of particles).
    """

    def __init__(self, record_rejected: bool = False, ok: bool = True):
        self._particles = []
        self.record_rejected = record_rejected
        self.ok = ok

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
        return sum((particle.accepted_sum_stats + particle.rejected_sum_stats
                    for particle in self._particles), [])

    def first_m_sum_stats(self, m):
        """
        Get (at most) the first `m` summary statistics.

        Returns
        -------

        sum_stats: List
            Concatenation of all the all_sum_stats lists of the first <= m
            particles added and accepted to this sample via append().
        """
        m = min(len(self._particles), m)

        return sum((particle.accepted_sum_stats + particle.rejected_sum_stats
                    for particle in self._particles[:m]), [])

    def first_m_particles(self, m) -> List:
        m = min(len(self._particles), m)

        return self._particles[:m]

    @property
    def _accepted_particles(self) -> List[Particle]:
        """
        Returns
        -------

        List of only the accepted particles.
        """
        return [particle for particle in self._particles if particle.accepted]

    def append(self, particle: Particle):
        """
        Add new particle to the sample.


        Parameters
        ----------

        particle: Particle
            Sampled particle containing all information needed later.
        """

        # add to population if accepted
        if particle.accepted or self.record_rejected:
            self._particles.append(particle)

    def __add__(self, other: "Sample"):
        sample = Sample(self.record_rejected)
        # sample's list of particles is the concatenation of both samples'
        # lists
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

    record_rejected: bool
        Corresponds to Sample.record_rejected.
    """

    def __init__(self, record_rejected: bool = False):
        self.record_rejected = record_rejected

    def __call__(self):
        """
        Create a new empty sample.
        """
        return Sample(self.record_rejected)


def wrap_sample(f):
    """
    Wrapper for Sampler.sample_until_n_accepted.
    Checks whether the sampling output is valid.
    """
    def sample_until_n_accepted(
            self, n, simulate_one, max_eval=np.inf, all_accepted=False):
        sample = f(self, n, simulate_one, max_eval, all_accepted)
        if sample.n_accepted != n and sample.ok:
            # this should not happen if the sampler is configured correctly
            raise AssertionError(
                f"Expected {n} but got {sample.n_accepted} acceptances.")
        return sample
    return sample_until_n_accepted


class SamplerMeta(ABCMeta):
    """
    This metaclass handles the checking of sampling output values.
    """

    def __init__(cls, name, bases, attrs):
        ABCMeta.__init__(cls, name, bases, attrs)
        cls.sample_until_n_accepted = wrap_sample(cls.sample_until_n_accepted)


class Sampler(ABC, metaclass=SamplerMeta):
    """
    Abstract Sampler base class.

    Produce valid particles: :class:`pyabc.parameters.ValidParticle`.

    Attributes
    ----------
    nr_evaluations_: int
        This is set after a population and counts the total number
        of model evaluations. This can be used to calculate the acceptance
        rate.
    sample_factory: SampleFactory
        A factory to create empty samples.
    show_progress: bool
        Whether to show progress within a generation.
        Some samplers support this by e.g. showing a progress bar.
        Set via
        >>> sampler = Sampler()
        >>> sampler.show_progress = True
    """

    def __init__(self):
        self.nr_evaluations_: int = 0
        self.sample_factory: SampleFactory = \
            SampleFactory(record_rejected=False)
        self.show_progress: bool = False

    def _create_empty_sample(self) -> Sample:
        return self.sample_factory()

    @abstractmethod
    def sample_until_n_accepted(
            self,
            n: int,
            simulate_one: Callable,
            max_eval: int = np.inf,
            all_accepted: bool = False) -> Sample:
        """
        Performs the sampling, i.e. creation of a new generation (i.e.
        population) of particles.

        Parameters
        ----------
        n:
            The number of samples to be accepted. I.e. the population size.
        simulate_one:
            A function which internally performs the whole process of
            sampling parameters, simulating data, and comparing to observed
            data to check for acceptance, as indicated via the
            particle.accepted flag.
        max_eval:
            Maximum number of evaluations to perform. Some samplers can check
            this condition directly and can thus terminate proactively.
        all_accepted:
            If it is known in advance that all sampled particles will have
            particle.accepted == True, then setting all_accepted = True can
            reduce the computational overhead for dynamic schedulers. This
            is usually in particular the case in the initial calibration
            iteration.

        Returns
        -------
        sample: :class:`pyabc.sampler.Sample`
            The generated sample, which contains the new population.
        """
