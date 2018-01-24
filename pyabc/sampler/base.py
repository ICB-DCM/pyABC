from abc import ABC, abstractmethod
from typing import Callable, TypeVar, List
from ..parameters import ValidParticle

A = TypeVar('A')
B = TypeVar('B')


class Sampler(ABC):
    """
    Abstract Sampler base class.

    Produce valid particles: :class:`pyabc.parameters.ValidParticle`.

    Attributes
    ----------

    nr_evaluations_: int
        This is set after a population and counts the total number
        of model evaluations. This can be used to calculate the acceptance
        rate.
    """
    def __init__(self):
        self.nr_evaluations_ = 0

    @abstractmethod
    def sample_until_n_accepted(self, sample_one: Callable[[], A],
                                simulate_one: Callable[[A], ValidParticle],
                                accept_one: Callable[[ValidParticle], bool],
                                n: int) -> List[B]:
        """
        Parameters
        ----------
        sample_one: Callable[[], A]
            A function which takes no arguments and returns
            a proposal parameter :math:`\\theta`.

        simulate_one: Callable[[A], ValidParticle]
            A function which takes as sole argument a proposal
            parameter :math:`\\theta` as returned by `sample_one`.
            It returns the summary statistics :math:`s`.

        accept_one: Callable[[ValidParticle], bool]
            A function which takes as sole argument the summary
            statistics :math:`s` as returned by `simulate_one`.
            It returns `True` it the simulated sample is accepted
            and `False` otherwise.

        n: int
            The number of samples to be accepted. I.e. the population size.

        Returns
        -------
        valid_particles: List[:class:`pyabc.parameters.ValidParticle`]
            The list of accepted particles.
        """
