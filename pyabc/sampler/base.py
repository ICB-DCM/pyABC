from abc import ABC, ABCMeta, abstractmethod
import numpy as np
from numbers import Real
from typing import Callable, List, Union

from ..population import Particle, Population
from ..util import AnalysisVars
from ..distance import Distance
from ..epsilon import Epsilon
from ..acceptor import Acceptor


class Sample:
    """
    A Sample is created and filled during the sampling process by the Sampler.

    Parameters
    ----------
    record_rejected:
        Whether to record rejected particles as well, along with accepted
        ones.
    is_look_ahead:
        Whether this sample consists of particles generated in look-ahead mode.
    ok:
        Whether the sampling process succeeded (usually in generating the
        requested number of particles).
    """

    def __init__(self, record_rejected: bool = False,
                 is_look_ahead: bool = False,
                 ok: bool = True):
        self.particles: List[Particle] = []
        self.record_rejected: bool = record_rejected
        self.is_look_ahead: bool = is_look_ahead
        self.ok: bool = ok

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
        return [particle.sum_stat for particle in self.particles]

    def first_m_sum_stats(self, m):
        """
        Get (at most) the first `m` summary statistics.

        Returns
        -------

        sum_stats: List
            Concatenation of all the all_sum_stats lists of the first <= m
            particles added and accepted to this sample via append().
        """
        m = min(len(self.particles), m)
        return [particle.sum_stat for particle in self.particles[:m]]

    def first_m_particles(self, m) -> List:
        m = min(len(self.particles), m)

        return self.particles[:m]

    @property
    def accepted_particles(self) -> List[Particle]:
        """
        Returns
        -------

        List of only the accepted particles.
        """
        return [particle for particle in self.particles if particle.accepted]

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
            self.particles.append(particle)

    def __add__(self, other: "Sample"):
        sample = Sample(record_rejected=self.record_rejected)
        # sample's list of particles is the concatenation of both samples'
        # lists
        sample.particles = self.particles + other.particles
        # the other attributes may keep their defaults
        return sample

    @property
    def n_accepted(self) -> int:
        """
        Returns
        -------

        n_accepted: int
            Number of accepted particles.
        """
        return len(self.accepted_particles)

    def get_accepted_population(self) -> Population:
        """
        Returns
        -------

        population: Population
            A population of only the accepted particles.
        """
        return Population(self.accepted_particles)


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

    def __call__(self, is_look_ahead: bool = False):
        """
        Create a new empty sample.
        """
        return Sample(
            record_rejected=self.record_rejected, is_look_ahead=is_look_ahead)


def wrap_sample(f):
    """Wrapper for Sampler.sample_until_n_accepted.
    Checks whether the sampling output is valid.
    """
    def sample_until_n_accepted(self, n, simulate_one, t, **kwargs):
        sample = f(self, n, simulate_one, t, **kwargs)
        if sample.n_accepted != n and sample.ok:
            # this should not happen if the sampler is configured correctly
            raise AssertionError(
                f"Expected {n} but got {sample.n_accepted} acceptances.")
        if any(particle.preliminary for particle in sample.particles):
            raise AssertionError(
                "There cannot be non-evaluated particles.")
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
    analysis_id: str
        A universal unique id of the analysis, automatically generated by the
        inference routine.
    """

    def __init__(self):
        self.nr_evaluations_: int = 0
        self.sample_factory: SampleFactory = \
            SampleFactory(record_rejected=False)
        self.show_progress: bool = False
        self.analysis_id: Union[str, None] = None

    def _create_empty_sample(self) -> Sample:
        return self.sample_factory()

    def set_analysis_id(self, analysis_id: str):
        """Set the analysis id.
        Called by the inference routine.
        The default is to just obediently set it. Specific samplers may want to
        check whether there are conflicting analyses.
        """
        self.analysis_id = analysis_id

    @abstractmethod
    def sample_until_n_accepted(
        self,
        n: int,
        simulate_one: Callable,
        t: int,
        *,
        max_eval: Real = np.inf,
        all_accepted: bool = False,
        ana_vars: AnalysisVars = None,
    ) -> Sample:
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
        t:
            Generation index for which to sample.
        max_eval:
            Maximum number of evaluations to perform. Some samplers can check
            this condition directly and can thus terminate proactively.
        all_accepted:
            If it is known in advance that all sampled particles will have
            particle.accepted == True, then setting all_accepted = True can
            reduce the computational overhead for dynamic schedulers. This
            is usually in particular the case in the initial calibration
            iteration.
        ana_vars:
            Various analysis variables. Some samplers can use these e.g. for
            proactive sampling.

        Returns
        -------
        sample: :class:`pyabc.sampler.Sample`
            The generated sample, which contains the new population.
        """

    def stop(self) -> None:
        """Stop the sampler.
        Called by the inference routine when an analysis is finished.
        Some samplers may need to e.g. finish ongoing processes or close
        servers.
        """

    def check_analysis_variables(
            self,
            distance_function: Distance,
            eps: Epsilon,
            acceptor: Acceptor) -> None:
        """Raise if any analysis variable is not conform with the sampler.
        This check serves in particular to ensure that all components are fit
        for look-ahead sampling. Default: Do nothing.
        """
