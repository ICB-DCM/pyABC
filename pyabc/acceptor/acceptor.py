"""
Acceptor
--------

After summary statistics for samples for given parameters have
been generated, it must be checked whether these are to be
accepted or not. This happens in the Acceptor class.

The most typical and simple way is to compute the distance
between simulated and observed summary statistics, and accept
if this distance is below some epsilon threshold. However, also
more complex acceptance criteria are possible, in particular
when the distance measure and epsilon criteria develop over
time.
"""

import numpy as np
import pandas as pd
from typing import Callable, List, Union
import logging

from ..distance import Distance, StochasticKernel, RET_SCALE_LIN
from ..epsilon import Epsilon
from pyabc import Parameter
from .temperature_scheme import (scheme_acceptance_rate,
                                 scheme_exponential_decay)
from .pdf_max_eval import pdf_max_take_from_kernel


logger = logging.getLogger("Acceptor")


class AcceptanceResult(dict):
    """
    Result of an acceptance step.

    Parameters
    ----------

    distance: float
        Distance value obtained.
    accept: bool
        A flag indicating the recommendation
        to accept or reject. More specifically:
        True: The distance is below the epsilon threshold.
        False: The distance is above the epsilon threshold.
    weight: float, optional (default = 1.0)
        Weight associated with the evaluation, which may need
        to be taken into account via importane sampling in
        calculating the parameter weight.
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

        An AcceptanceResult.


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

    def get_epsilon_equivalent(self, t: int):
        """
        Return acceptance criterion for time t. An acceptor should implement
        this if it manages the acceptance criterion itself, i.e. when it is
        used together with a NoEpsilon.
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

    def __init__(self, fun):
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


def accept_uniform_use_current_time(
        distance_function, eps, x, x_0, t, par):
    """
    Use only the distance function and epsilon criterion at the current time
    point to evaluate whether to accept or reject.
    """
    d = distance_function(x, x_0, t, par)
    accept = d <= eps(t)

    return AcceptanceResult(d, accept)


def accept_uniform_use_complete_history(
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

    return AcceptanceResult(d, accept)


class UniformAcceptor(Acceptor):
    """
    Base acceptance on the distance function and a uniform error distribution
    between -eps and eps.
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
            return accept_uniform_use_complete_history(
                distance_function, eps, x, x_0, t, par)
        else:  # use only current time
            return accept_uniform_use_current_time(
                distance_function, eps, x, x_0, t, par)


class StochasticAcceptor(Acceptor):
    """
    This acceptor implements a stochastic acceptance step based on a
    probability density, generalizing from the uniform acceptance kernel.
    A particle is accepted if for the simulated summary statistics x
    and the observed summary statistics x_0 holds

    .. math::

       \\frac{\\text{pdf}(x_0|x)}{c}\\geq u

    where u ~ U[0,1], and c is a normalizing constant.

    The concept is based on [#wilkinson]_.

    .. [#wilkinson] Wilkinson, Richard David; "Approximate Bayesian
        computation (ABC) gives exact results under the assumption of model
        error"; Statistical applications in genetics and molecular biology
        12.2 (2013): 129-141.

    """

    def __init__(
            self,
            temp_schemes: Union[Callable, List[Callable]] = None,
            pdf_max_method: Callable = None,
            **kwargs):
        """
        Parameters
        ----------

        pdf: callable, optional
            A probability density function

            .. math::

               pdf(x_0|x)

            of the observed summary statistics given the simulated
            summary statistics. If None is passed, a standard multivariate
            normal distribution is assumed.

        temp_schemes: Union[Callable, List[Callable]], optional
            Temperature schemes of the form
            Callable[[dict, **kwargs], float]
            returning proposed temperatures for the next time point. If
            multiple are passed, the minimum computed temperature is used.
            If the next time point is the last time point according to
            max_nr_populations, 1.0 is used for exact inference.

        pdf_max_method: Callable, optional
            Method how to compute the normalization constant c.
            The normalization value the density is divided by. To have
            acceptance from the desired distribution, c should be
            at least (and as precisely as possible for higher acceptance
            rates) the highest mode of the distribution.
            If None is passed, it is computed, assumed to be for x=x_0.

        kwargs: dict, optional
            Passed to the schedulers. Supported arguments that have a default
            value:
            * target_acceptance_rate: float = 0.5: target acceptance rate
            * temp_init: float = None (i.e. estimated from prior): initial
              temperature
            * temp_decay_exponent: float = 3: Exponent with which the
              temperature decays in fixed-decay schemes.
            * config: dict: Can be used as a memory object.

            In addition, the schedulers receive time-specific info, see the
            _update() method for details.
        """

        super().__init__()

        if temp_schemes is None:
            temp_schemes = [scheme_acceptance_rate, scheme_exponential_decay]
        elif not isinstance(temp_schemes, list):
            temp_schemes = [temp_schemes]
        self.temp_schemes = temp_schemes

        if pdf_max_method is None:
            pdf_max_method = pdf_max_take_from_kernel
        self.pdf_max_method = pdf_max_method

        # default kwargs
        default_kwargs = dict(
            temp_init=None,
            config={}
        )
        # set kwargs to default if not specified
        for key, value in default_kwargs.items():
            kwargs.setdefault(key, value)
        self.kwargs = kwargs

        # maximum pdfs, indexed by time
        self.pdf_maxs = {}

        # temperatures, indexed by time
        self.temperatures = {}

        # fields to be filled later
        self.x_0 = None
        self.max_nr_populations = None

    def initialize(
            self,
            t: int,
            get_weighted_distances: Callable[[], pd.DataFrame],
            max_nr_populations: int,
            distance_function: Distance,
            x_0: dict):
        """
        Initialize temperature and maximum pdf.
        """
        self.x_0 = x_0
        self.max_nr_populations = max_nr_populations

        if not isinstance(distance_function, StochasticKernel):
            raise AssertionError(
                "The distance function must be a pyabc.StochasticKernel.")

        # update
        self._update(t, get_weighted_distances, distance_function, 1.0)

    def update(self,
               t: int,
               weighted_distances: pd.DataFrame,
               distance_function: Distance,
               acceptance_rate: float):
        self._update(
            t, lambda: weighted_distances, distance_function, acceptance_rate)

    def _update(self,
                t: int,
                get_weighted_distances: Callable[[], pd.DataFrame],
                kernel: Distance,
                acceptance_rate: float):
        """
        Update schemes for the upcoming time point t.
        """
        # update pdf_max

        pdf_max = self.pdf_max_method(
            kernel_val=kernel.pdf_max,
            get_weighted_distances=get_weighted_distances,
            pdf_maxs=self.pdf_maxs)
        self.pdf_maxs[t] = pdf_max

        logger.debug(f"acceptance rate={acceptance_rate}")
        logger.debug(f"pdf_max={self.pdf_maxs[t]} for t={t}.")

        # update temperature

        if t >= self.max_nr_populations - 1:
            # t is last time
            self.temperatures[t] = 1.0
        else:
            # evaluate schedulers
            temps = []
            for scheme in self.temp_schemes:
                temp = scheme(
                    t=t,
                    get_weighted_distances=get_weighted_distances,
                    x_0=self.x_0,
                    pdf_max=self.pdf_maxs[t],
                    ret_scale=kernel.ret_scale,
                    temperatures=self.temperatures,
                    max_nr_populations=self.max_nr_populations,
                    acceptance_rate=acceptance_rate,
                    **self.kwargs)
                temps.append(temp)

            logger.debug(f"Proposed temperatures: {temps}.")

            # take reasonable minimum temperature
            fallback = self.temperatures[t - 1] \
                if t - 1 in self.temperatures else np.inf
            temp = max(min(*temps, fallback), 1.0)

            # fill into temperatures list
            self.temperatures[t] = temp

    def __call__(self, distance_function, eps, x, x_0, t, par):
        # rename
        kernel = distance_function

        # temperature
        temp = self.temperatures[t]

        # compute probability density
        pd = kernel(x, x_0, t, par)
        pdf_max = self.pdf_maxs[t]

        # check pdf max ok
        if pdf_max < pd:
            logger.info(
                f"Encountered a density {pd} > current pdf_max {pdf_max}.")

        # rescale
        if kernel.ret_scale == RET_SCALE_LIN:
            pd_rescaled = pd / self.pdf_maxs[t]
        else:  # kernel.ret_scale == RET_SCALE_LOG
            pd_rescaled = np.exp(pd - self.pdf_maxs[t])

        # acceptance probability
        acceptance_probability = pd_rescaled ** (1 / temp)

        # accept
        threshold = np.random.uniform(low=0, high=1)
        if acceptance_probability >= threshold:
            accept = True
        else:
            accept = False

        # return unscaled density value and the acceptance flag
        return AcceptanceResult(pd, accept)

    def get_epsilon_equivalent(self, t: int):
        return self.temperatures[t]
