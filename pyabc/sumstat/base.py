"""Summary statistics."""

from abc import ABC, abstractmethod
from typing import Callable, Union


class Sumstat(ABC):
    """Abstract base class for summary statistics."""

    @abstractmethod
    def __call__(self, x):
        """Calculate summary statistics from raw model output."""


class IdentitySumstat(Sumstat):
    """Identity summary statistics.

    Asssumes that the model already returns the summary statistics, and that
    these are compatible with pyABC's assumptions (basically a dictionary of
    numpy arrays).
    """

    def __call__(self, x):
        return x


class FunctionSumstat(Sumstat):
    """Wrapper around a function calculating the summary statistics."""

    def __init__(self, fun):
        self.fun = fun

    def __call__(self, x):
        return self.fun(x)


def to_sumstat(maybe_sumstat: Union[Callable, Sumstat]) -> Sumstat:
    """Keep sumstats, convert callables to sumstats.

    Parameters
    ----------
    maybe_sumstat: A callable or sumstat instance

    Returns
    -------
    sumstat: A valid sumstat instance.
    """
    if isinstance(maybe_sumstat, Sumstat):
        return maybe_sumstat
    return FunctionSumstat(maybe_sumstat)
