"""
Acceptor
--------

After summary statistics of samples for given parameters have
been generated, it must be checked whether these are to be
accepted or not. This happens in the Acceptor class.

The most typical and simple way is to compute the distance
between simulated and observed summary statistics, and accept
if this distance is below some epsilon threshold. However, also
more complex acceptance criteria are possible, in particular
when the distance measure and epsilon criteria develop over
time.
"""

import pandas as pd
from typing import Callable
import logging

from ..distance import Distance
from ..epsilon import Epsilon
from ..parameters import Parameter


logger = logging.getLogger("Acceptor")


class AcceptorResult(dict):
    """
    Result of an acceptance step.

    Parameters
    ----------

    distance: float
        Distance value obtained.
    accept: bool
        A flag indicating the recommendation to accept or reject.
        More specifically:
        True: The distance is below the acceptance threshold.
        False: The distance is above the acceptance threshold.
    weight: float, optional (default = 1.0)
        Weight associated with the evaluation, which may need
        to be taken into account via importance sampling when
        calculating the parameter weight. Defaults to 1.0.
    """

    def __init__(self, distance: float, accept: bool, weight: float = 1.0):
        super().__init__()
        self.distance = distance
        self.accept = accept
        self.weight = weight

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Acceptor:
    """
    The acceptor class encodes the acceptance step.
    This class is abstract and cannot be invoked itself.
    """

    def __init__(self):
        """
        Default constructor.
        """
        pass

    def initialize(
            self,
            t: int,
            get_weighted_distances: Callable[[], pd.DataFrame],
            max_nr_populations: int,
            distance_function: Distance,
            x_0: dict):
        """
        Initialize. This method is called by the ABCSMC framework initially,
        and can be used to calibrate the acceptor to initial statistics.
        The default is to do nothing.

        Parameters
        ----------

        t: int
            The timepoint to initialize the acceptor for.
        get_weighted_distances: Callable[[], pd.DataFrame]
            Returns on demand the distances for initializing the acceptor.
        max_nr_populations: int
            Maximum number of populations in current run.
        distance_function: Distance
            Distance object. The acceptor should not modify it, but might
            extract some meta information.
        x_0: dict
            The observed summary statistics.
        """
        pass

    def update(self,
               t: int,
               weighted_distances: pd.DataFrame,
               distance_function: Distance,
               acceptance_rate: float):
        """
        Update the acceptance criterion.

        Parameters
        ----------

        t: int
            The timepoint to initialize the acceptor for.
        weighted_distances: Callable[[], pd.DataFrame]
            The current generation's weighted distances.
        distance_function: Distance
            Distance object.
        acceptance_rate: float
            The current generation's acceptance rate.
        """
        pass

    def __call__(self,
                 distance_function: Distance,
                 eps: Epsilon,
                 x: dict, x_0: dict,
                 t: int,
                 par: Parameter):
        """
        Compute distance between summary statistics and evaluate whether to
        accept or reject.
        All concrete implementations must implement this method.

        Parameters
        ----------

        distance_function: pyabc.Distance
            The distance function.
            The user is free to use or ignore this function.
        eps: pyabc.Epsilon
            The acceptance thresholds.
            The user is free to use or ignore this object.
        x: dict
            Current summary statistics to evaluate.
        x_0: dict
            The observed summary statistics.
        t: int
            Time point for which to check.
        par: pyabc.Parameter
            The model parameters used to simulate x.

        Returns
        -------

        An AcceptorResult.

        .. note::
            Currently, only one value encoding the distance is returned
            (and stored in the database),
            namely that at time t, even if also other distances affect the
            acceptance decision, e.g. distances from previous iterations,
            or in general if the distance is not scalar.
            This is because the last distance is likely to be most informative
            for the further process, and because in some parts of ABC a scalar
            distance value is required.
        """
        raise NotImplementedError()


class SimpleFunctionAcceptor(Acceptor):
    """
    Initialize from function.

    Parameters
    ----------

    fun: Callable, optional
        Callable with the same signature as the __call__ method.
    """

    def __init__(self, fun=None):
        super().__init__()
        self.fun = fun

    def __call__(self, distance_function, eps, x, x_0, t, par):
        return self.fun(distance_function, eps, x, x_0, t, par)

    @staticmethod
    def assert_acceptor(maybe_acceptor):
        """
        Create an acceptor object from input.

        Parameters
        ----------

        maybe_acceptor: Acceptor or Callable
            Either pass a full acceptor, or a callable which is then filled
            into a SimpleAcceptor.

        Returns
        -------

        acceptor: Acceptor
            An Acceptor object in either case.
        """
        if isinstance(maybe_acceptor, Acceptor):
            return maybe_acceptor
        else:
            return SimpleFunctionAcceptor(maybe_acceptor)


def accept_use_current_time(
        distance_function, eps, x, x_0, t, par):
    """
    Use only the distance function and epsilon criterion at the current time
    point to evaluate whether to accept or reject.
    """

    d = distance_function(x, x_0, t, par)
    accept = d <= eps(t)

    return AcceptorResult(distance=d, accept=accept)


def accept_use_complete_history(
        distance_function, eps, x, x_0, t, par):
    """
    Use the acceptance criteria from the complete history to evaluate whether
    to accept or reject.

    This includes time points 0,...,t, as far as these are
    available. If either the distance function or the epsilon criterion cannot
    handle any time point in this interval, the resulting error is simply
    intercepted and the respective time not used for evaluation. This situation
    can frequently occur when continuing a stopped run. A different behavior
    is easy to implement.
    """
    # first test current criterion, which is most likely to fail
    d = distance_function(x, x_0, t, par)
    accept = d <= eps(t)

    if accept:
        # also check against all previous distances and acceptance criteria
        for t_prev in range(0, t):
            try:
                d_prev = distance_function(x, x_0, t_prev, par)
                accept = d_prev <= eps(t_prev)
                if not accept:
                    break
            except Exception:
                # ignore as of now
                accept = True

    return AcceptorResult(distance=d, accept=accept)


class UniformAcceptor(Acceptor):
    """
    Base acceptance on the distance function and a uniform error distribution
    between -eps and +eps.
    This is the most common acceptance criterion in ABC.
    """

    def __init__(self, use_complete_history: bool = False):
        """
        Parameters
        ----------

        use_complete_history: bool, optional
            Whether to compare to all previous distances and epsilons, or use
            only the current distance time (default). This can be of interest
            with adaptive distances, in order to guarantee nested acceptance
            regions.
        """
        super().__init__()
        self.use_complete_history = use_complete_history

    def __call__(self, distance_function, eps, x, x_0, t, par):
        if self.use_complete_history:
            return accept_use_complete_history(
                distance_function, eps, x, x_0, t, par)
        else:  # use only current time
            return accept_use_current_time(
                distance_function, eps, x, x_0, t, par)
