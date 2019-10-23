import numpy as np
import scipy as sp
import pandas as pd
from typing import Callable, List, Union
import logging

from .base import Epsilon
from ..distance import SCALE_LIN

logger = logging.getLogger("Temperature")


class Temperature(Epsilon):
    """
    A temperatur scheme handles the decrease of the temperatures employed
    by a :class:`pyabc.acceptor.StochasticAcceptor` over time.

    Parameters
    ----------
    schemes: Union[Callable, List[Callable]]
        Temperature schemes of the form
        ``Callable[[dict, **kwargs], float]`` returning proposed
        temperatures for the next time point.
    aggregate_fun: Callable[List[int], int]
        The function to aggregate the schemes by, of the form
        ``Callable[List[float], float]``.
        Defaults to taking the minimum.
    initial_temperature: float
        The initial temperature. If None provided, an AcceptanceRateScheme
        is used.
    maximum_nr_populations: int
        The maximum number of iterations as passed to ABCSMC.
        May be inf.
    temperatures: Dict[int, float]
        The temperatures, format key:temperature.
    """

    def __init__(
            self,
            schemes: Union[Callable, List[Callable]] = None,
            aggregate_fun: Callable[List[int], int] = None,
            initial_temperature: float = None):
        if schemes is None:
            schemes = [AcceptanceRateScheme(), ExponentialDecayScheme()]
        self.schemes = schemes

        if aggregate_fun is None:
            aggregate_fun = min
        self.aggregate_fun = aggregate_fun

        self.initial_temperature = initial_temperature

        self.max_nr_populations = None
        self.temperatures = {}

    def initialize(self,
                   t: int,
                   get_weighted_distances: Callable[[], pd.DataFrame],
                   max_nr_populations: int,
                   acceptor_config: dict):
        self.max_nr_populations = max_nr_populations
        self._update(t, get_weighted_distances, 1.0, acceptor_config)

    def update(self,
               t: int,
               weighted_distances: pd.DataFrame,
               acceptance_rate: float,
               acceptor_config: dict):
        self._update(t, lambda: weighted_distances, acceptance_rate,
                     acceptor_config)

    def _update(self,
                t: int,
                get_weighted_distances: Callable[[], pd.DataFrame],
                acceptance_rate: float,
                acceptor_config):
        """
        Compute the temperature for time `t`.
        """
        # update the temperature

        # scheme arguments
        kwargs = dict(
            t=t,
            get_weighted_distances=get_weighted_distances,
            max_nr_populations=self.max_nr_populations,
            pdf_norm=acceptor_config['pdf_norm'],
            kernel_scale=acceptor_config['kernel_scale'],
            prev_temperature=self.temperatures.get(t - 1, None),
            acceptance_rate=acceptance_rate,
        )

        if t >= self.max_nr_populations - 1:
            # t is last time
            temperature = 1.0
        elif not self.temperatures and self.initial_temperature is not None:
            if callable(self.initial_temperature):
                # execute scheme
                temperature = self.initial_temperature(**kwargs)
            else:
                # is float value
                temperature = self.initial_temperature
        else:
            # evalute schemes
            temps = []
            for scheme in self.schemes:
                temp = scheme(**kwargs)
                temps.append(temp)
            logger.debug(f"Proposed temperatures: {temps}.")

            # compute next temperature based on proposals and fallback
            # should not be higher than before
            fallback = self.temperatures[t - 1] \
                if t - 1 in self.temperatures else np.inf
            proposed_value = self.aggregate_fun(temps)
            # also a value lower than 1.0 does not make sense
            temperature = max(min(proposed_value, fallback), 1.0)

        # record found value
        self.temperatures[t] = temperature

    def __call__(self,
                 t: int) -> float:
        return self.temperatures[t]


class TemperatureScheme:

    def __init__(self):
        pass

    def __call__(self,
                 t: int,
                 get_weighted_distances: Callable,
                 max_nr_populations: int,
                 pdf_norm: float,
                 kernel_scale: str,
                 prev_temperature: float,
                 acceptance_rate: float):
        pass


class AcceptanceRateScheme(TemperatureScheme):
    """
    Try to keep the acceptance rate constant at a value of
    `target_acceptance_rate`. Note that this scheme will fail to
    reduce the temperature sufficiently in later iterations, if the
    problem's inherent acceptance rate is lower, but it has been
    observed to give big temperature leaps in early iterations.

    Parameters
    ----------
    target_rate: float
        The target acceptance rate to match.
    """

    def __init__(self, target_rate: float = 0.5):
        self.target_rate = target_rate

    def __call__(self,
                 t: int,
                 get_weighted_distances: Callable,
                 max_nr_populations: int,
                 pdf_norm: float,
                 kernel_scale: str,
                 prev_temperature: float,
                 acceptance_rate: float):
        # execute function ( expensive if in calibration)
        df = get_weighted_distances()

        weights = np.array(df['w'])
        pdfs = np.array(df['distance'])

        # compute rescaled posterior densities
        if kernel_scale == SCALE_LIN:
            values = pdfs / pdf_norm
        else:  # kernel_scale == SCALE_LOG
            values = np.exp(pdfs - pdf_norm)

        # to acceptance probabilities
        values = np.minimum(values, 1.0)

        # to pmf
        # TODO: We currently use the weights from the previous iteration.
        # It might be better to use weights based on `transition / prior`
        # for the next proposal transition kernel.
        weights /= np.sum(weights)

        # objective function which we wish to find a root for
        def obj(beta):
            val = np.sum(weights * values**beta) - self.arget_rate
            return val

        if obj(1) > 0:
            # function is mon. dec., smallest possible value already > 0
            beta_opt = 1.0
            # it is obj(0) >= 0 always
        else:
            # perform binary search
            # TODO: check out more efficient optimization approach
            beta_opt = sp.optimize.bisect(obj, 0, 1)

        # temperature is inverse beta
        temperature = 1.0 / beta_opt
        return temperature


class ExponentialDecayScheme(TemperatureScheme):
    """
    If `max_nr_populations` is finite:

    .. math::
        T_j = T_{max}^{(n-j)/n}

    where n denotes the number of populations, and j=1,...,n the iteration.
    This translates to

    .. math::
        T_j = T_{j-1}^{(n-j)/(n-(j-1))}.

    This ensures that a temperature of 1.0 is reached after exactly the
    remaining number of steps.

    If `max_nr_populations` is infinite, the next temperature is

    .. math::
        T_j = \\alpha \\cdot T_{j-1},

    where by default alpha=0.25.

    So, in both cases the sequence of temperatures follows an exponential
    decay, also known as a geometric progression, or a linear progression
    in log-space.

    Note that the formula is applied anew in each iteration. That has the
    advantage that, if also other schemes are used s.t. T_{j-1} is smaller
    than by the above, advantage can be made of this.

    Parameters
    ----------

    alpha: float
        Factor by which to reduce the temperature, if `max_nr_populations`
        is infinite.
    """

    def __init__(self, alpha: float = 0.25):
        self.alpha = alpha

    def __call__(self,
                 t: int,
                 get_weighted_distances: Callable,
                 max_nr_populations: int,
                 pdf_norm: float,
                 kernel_scale: str,
                 prev_temperature: float,
                 acceptance_rate: float):
        # needs a starting temperature
        # if not available, return infinite temperature
        if prev_temperature is None:
            return np.inf

        # base temperature
        temp_base = prev_temperature

        if max_nr_populations == np.inf:
            # just decrease by a factor of alpha each round
            temperature = self.alpha * temp_base
            return temperature

        # how many steps left?
        t_to_go = max_nr_populations - t

        # compute next temperature according to exponential decay
        temperature = temp_base ** ((t_to_go - 1) / t_to_go)

        return temperature


class DalyScheme(TemperatureScheme):
    """
    This scheme is loosely based on [#daly2017]_, however note that it does
    not try to replicate it entirely. In particular, the implementation
    of pyABC does not allow the sampling to be stopped when encountering
    too low acceptance rates, such that this can only be done ex-posteriori
    here.

    .. [#daly2017] Daly Aidan C., Cooper Jonathan, Gavaghan David J. ,
            and Holmes Chris. "Comparing two sequential Monte Carlo samplers
            for exact and approximate Bayesian inference on biological
            models". Journal of The Royal Society Interface, 2017
    """

    def __init__(self, alpha: float = 0.5, min_rate: float = 1e-4):
        self.alpha = alpha
        self.min_rate = min_rate
        self.k = {}

    def __call__(self,
                 t: int,
                 get_weighted_distances: Callable,
                 max_nr_populations: int,
                 pdf_norm: float,
                 kernel_scale: str,
                 prev_temperature: float,
                 acceptance_rate: float):
        # needs a starting temperature
        # if not available, return infinite temperature
        if prev_temperature is None:
            return np.inf

        # base temperature
        temp_base = prev_temperature

        # addressing the std, not the var
        eps_base = np.sqrt(temp_base)

        if not self.k:
            # initial iteration
            self.k[t - 1] = eps_base

        k_base = self.k[t - 1]

        if acceptance_rate < self.min_rate:
            logger.debug("DalyScheme: Reacting to low acceptance rate.")
            # reduce reduction
            k_base = self.alpha * k_base

        self.k[t] = min(k_base, self.alpha * eps_base)
        eps = eps_base - self.k[t]
        temperature = eps**2

        return temperature
