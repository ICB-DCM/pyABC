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
import scipy as sp
from typing import Callable, List, Union
import logging


logger = logging.getLogger("Acceptor")


class Acceptor:
    """
    This class encodes the acceptance step.
    """

    def __init__(self):
        """
        Default constructor.
        """

    def initialize(self,
                   t: int,
                   get_sum_stats: Callable[[], List[dict]],
                   max_nr_populations: int,
                   x_0: dict):
        """
        Initialize. This method is called by the ABCSMC framework initially,
        and can be used to calibrate the acceptor to initial statistics.

        The defaul implementation is to do nothing.
        """
        pass

    def update(self,
               t: int,
               sum_stats: List[dict]):
        """
        Update the acceptance criterion.
        """
        pass

    def __call__(self, t, distance_function, eps, x, x_0, pars):
        """
        Compute distance between summary statistics and evaluate whether to
        accept or reject.

        This class is abstract and cannot be used on its own. The simplest
        usable class is the derived SimpleAcceptor.

        Parameters
        ----------

        t: int
            Time point for which to check.

        distance_function: pyabc.DistanceFunction
            The distance function.
            The user is free to use or ignore this function.

        eps: pyabc.Epsilon
            The acceptance thresholds.

        x: dict
            Current summary statistics to evaluate.

        x_0: dict
            The observed summary statistics.

        pars: dict
            The model parameters.

        Returns
        -------

        (distance, accept): (float, bool)
            True: The distance function is below the epsilon threshold.
            False: The distance function is above the epsilon threshold.

        .. note::
            Currently, only one value encoding the distance is returned
            (and stored in the database),
            namely that at time t, even if also other distances affect the
            acceptance decision, e.g. distances from previous iterations. This
            is because the last distance is likely to be most informative for
            the further process.
        """
        raise NotImplementedError()

    def get_epsilon_equivalent(self, t: int):
        """
        Return acceptance criterion for time t. An acceptor should implement
        this if it manages the acceptance criterion itself, i.e. when it is
        used together with a NoEpsilon.
        """
        return np.inf


class SimpleAcceptor(Acceptor):
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

    def __call__(self, t, distance_function, eps, x, x_0, pars):
        return self.fun(t, distance_function, eps, x, x_0, pars)

    @staticmethod
    def assert_acceptor(acceptor):
        """
        Parameters
        ----------

        acceptor: Acceptor or Callable
            Either pass a full acceptor, or a callable which is then filled
            into a SimpleAcceptor.

        Returns
        -------

        acceptor: Acceptor
            An Acceptor object in either case.
        """
        if isinstance(acceptor, Acceptor):
            return acceptor
        else:
            return SimpleAcceptor(acceptor)


def accept_uniform_use_current_time(
        t, distance_function, eps, x, x_0, pars):
    """
    Use only the distance function and epsilon criterion at the current time
    point to evaluate whether to accept or reject.
    """

    d = distance_function(t, x, x_0)
    accept = d <= eps(t)

    return d, accept


def accept_uniform_use_complete_history(
        t, distance_function, eps, x, x_0, pars):
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
    d = distance_function(t, x, x_0)
    accept = d <= eps(t)

    if accept:
        # also check against all previous distances and acceptance criteria
        for t_prev in range(0, t):
            try:
                d_prev = distance_function(t_prev, x, x_0)
                accept = d_prev <= eps(t_prev)
                if not accept:
                    break
            except Exception:
                # ignore as of now
                accept = True

    return d, accept


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

    def __call__(self, t, distance_function, eps, x, x_0, pars):
        if self.use_complete_history:
            return accept_uniform_use_complete_history(
                t, distance_function, eps, x, x_0, pars)
        else:  # use only current time
            return accept_uniform_use_current_time(
                t, distance_function, eps, x, x_0, pars)


class StochasticAcceptor(Acceptor):
    """
    This acceptor implements a stochastic acceptance step based on a
    probability density, generalizing from the uniform acceptance kernel.
    A particle is accepted, if for the simulated summary statistics x,
    and the observed summary statistics x_0 holds

    .. math::

       \\frac{pdf(x_0|x)}{c}\\geq u

    where u ~ U[0,1], and c is a normalizing constant.

    The concept is based on [#wilkinson]_.

    .. [#wilkinson] Wilkinson, Richard David; "Approximate Bayesian
        computation (ABC) gives exact results under the assumption of model
        error"; Statistical applications in genetics and molecular biology
        12.2 (2013): 129-141.

    """

    def __init__(
            self,
            pdf=None,
            c=None,
            temp_schemes: Union[Callable, List[Callable]] = None,
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

        c: float, optional
            The normalization value the density is divided by. To have
            acceptance from the desired distribution, c should be
            at least (and as precisely as possible for higher acceptance
            rates) the highest mode of the distribution.
            If None is passed, it is computed, assumed to be for x=x_0.

        temp_schemes: Union[Callable, List[Callable]], optional
            Temperature schemes of the form 
            Callable[[dict, **kwargs], float]
            returning proposed temperatures for the next time point. If
            multiple are passed, the minimum computed temperature is used.
            If the next time point is the last time point according to
            max_nr_populations, 1.0 is used for exact inference.

        kwargs: dict, optional
            Passed to the schedulers. Supported arguments that have a default
            value:
            * target_acceptance_rate
            * temp_init
            * temp_decay_exponent
            * config: dict
            In addition, the schedulers receive time-specific info, see the
            _update() method for details.
        """

        super().__init__()

        self.pdf = pdf
        self.c = c

        if temp_schemes is None:
            temp_schemes = [scheme_acceptance_rate, scheme_decay]
        elif not isinstance(temp_schemes, list):
            temp_schemes = [temp_schemes]
        if not len(temp_schemes):
            raise ValueError("At least one temperature scheduling method "
                             "is required.")
        self.temp_schemes = temp_schemes
        
        # default kwargs
        default_kwargs = dict(
            target_acceptance_rate = 0.5,
            temp_init = None,
            temp_decay_exponent = 3,
            alpha = 0.5,
            config = {}
        )
        # set kwargs to default if not specified
        for key, value in default_kwargs.items():
            kwargs.setdefault(key, value)
        self.kwargs = kwargs

        # temepratures, indexed by time
        self.temperatures = {}

        # fields to be filled later
        self.x_0 = None
        self.max_nr_populations = None

    def initialize(self,
                   t: int,
                   get_sum_stats: Callable[[], List[dict]],
                   max_nr_populations: int,
                   x_0):
        """
        Initialize temperature.
        """
        self.x_0 = x_0
        self.max_nr_populations = max_nr_populations

        # init normalization c
        if self.c is None:
            self.c = self.pdf(self.x_0, self.x_0)
        
        # update
        self._update(t, get_sum_stats, 1.0)

    def update(self,
               t: int,
               sum_stats: List[dict],
               acceptance_rate: float):
        self._update(t, lambda: sum_stats, acceptance_rate)

    def _update(self,
                t: int,
                get_sum_stats: Callable[[], List[dict]],
                acceptance_rate: float):

        # check if final time point reached
        if t >= self.max_nr_populations - 1:
            self.temperatures[t] = 1.0
            return

        # evaluate schedulers
        temps = []
        for scheme in self.temp_schemes:
            temp = scheme(t=t,
                          get_sum_stats=get_sum_stats,
                          x_0=self.x_0,
                          pdf=self.pdf,
                          c=self.c,
                          temperatures=self.temperatures,
                          max_nr_populations=self.max_nr_populations,
                          acceptance_rate=acceptance_rate,
                          **self.kwargs)
            temps.append(temp)

        logger.debug(f"proposed temperatures: {temps}")
        logger.debug(f"acceptance_rate: {acceptance_rate}")

        # take reasonable minimum temperature
        temp = max(min(temps), 1.0)

        # fill into temperatures list
        self.temperatures[t] = temp


    def __call__(self, t, distance_function, eps, x, x_0, pars):
        # temperature
        temp = self.temperatures[t]

        # compute probability density
        pd = self.pdf(x_0, x)

        # rescale
        pd_rescaled = pd / self.c

        # acceptance probability
        acceptance_probability = pd_rescaled ** (1 / temp)

        # accept
        threshold = np.random.uniform(low=0, high=1)
        if acceptance_probability >= threshold:
            accept = True
        else:
            accept = False

        return pd_rescaled, accept

    def get_epsilon_equivalent(self, t: int):
        return self.temperatures[t]


def get_pdf_normal(mean, cov):
    """
    Generate a normal probability density function for given
    mean and covariance.
    """
    def pdf(x_0, x):
        v_0 = np.array(list(x_0.values()))
        v = np.array(list(x.values()))
        
        return sp.stats.multivariate_normal.pdf(v_0 - v, mean=mean, cov=cov)

    c = sp.stats.multivariate_normal.pdf(np.zeros(len(v)), mean=mean, cov=cov)

    return pdf, c


def get_pmf_binomial(p):
    """
    Generate a binomial probability mass function
    for a given success probability p.
    """
    def pdf(x_0, x):
        v_0 = np.array(list(x_0.values()))
        v = np.array(list(x.values()))

        pmf = 1
        for j in range(len(v)):
            pmf *= sp.stats.binom.pmf(k=v_0[j], n=v[j], p=p) \
                   if v[j] > 0 else 1

        return pmf

    c = 1
    # if we have a better bound on n, we can improve the guess of c

    return pdf, c


def scheme_acceptance_rate(**kwargs):
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    temp_init = kwargs['temp_init']
    get_sum_stats = kwargs['get_sum_stats']
    pdf = kwargs['pdf']
    c = kwargs['c']
    x_0 = kwargs['x_0']
    target_acceptance_rate = kwargs['target_acceptance_rate']

    # is there a pre-defined step to start with?
    if t - 1 not in temperatures and temp_init is not None:
        return temp_init

    # execute function
    sum_stats = get_sum_stats()

    # compute rescaled posterior densities
    values = [pdf(x_0, x) / c
              for x in sum_stats]
    values = np.array(values)

    # objective function which we wish to find a root for
    def obj(beta):
        val = np.sum(values**beta) / values.size - \
            target_acceptance_rate
        return val

    if obj(1) > 0:
        beta_opt = 1.0
    else:
        # perform binary search
        # TODO: take a more efficient optimization approach?
        beta_opt = sp.optimize.bisect(obj, 0, 1)

    # temperature is inverse beta
    temp_opt = 1.0 / beta_opt
    return temp_opt


def scheme_exponential_decay(**kwargs):
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    max_nr_populations = kwargs['max_nr_populations']
    temp_init = kwargs['temp_init']
    alpha = kwargs['alpha']

    if t - 1 in temperatures:
        temp_base = temperatures[t - 1]
    elif temp_init is not None:
        return temp_init
    else:
        return scheme_acceptance_rate(**kwargs)

    if max_nr_populations == np.inf:
        # just decrease by a factor of alpha each round
        temp = alpha * temp_base
        return tmp

    # how many steps left?
    t_to_go = (max_nr_populations - 1) - (t - 1)
    
    temp = temp_base ** ((t_to_go - 1) / t_to_go)
    return temp


def scheme_decay(**kwargs):
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    max_nr_populations = kwargs['max_nr_populations']
    temp_init = kwargs['temp_init']
    temp_decay_exponent = kwargs['temp_decay_exponent']

    # check if we can compute a decay step
    if max_nr_populations == np.inf:
        raise ValueError("Can only perform decay step with a finite "
                         "max_nr_populations.")

    # get temperature to start with
    if t - 1 in temperatures:
        temp_base = temperatures[t - 1]
    elif temp_init is not None:
        return temp_init
    else:
        return scheme_acceptance_rate(**kwargs)

    # how many steps left?
    t_to_go = (max_nr_populations - 1) - (t - 1)
    if t_to_go < 2:
        # have to take exact step, i.e. a temperature of 1, next
        return 1.0
    temps = np.linspace(1, (temp_base)**(1 / temp_decay_exponent),
                        t_to_go + 1) ** temp_decay_exponent
    logger.debug(f"temperatures: {temps}")

    # pre-last step is the next step
    temp = temps[-2]
    return temp


def scheme_daly(**kwargs):
    """
    Use modified scheme in daly 2017. 
    """
    # required fields
    t = kwargs['t']
    temperatures = kwargs['temperatures']
    temp_init = kwargs['temp_init']
    acceptance_rate = kwargs['acceptance_rate']
    min_acceptance_rate = kwargs.get('min_acceptance_rate', 1e-3)

    config = kwargs.get('config', {})
    k = config.setdefault('k', {t: temp_init})

    alpha = kwargs.get('alpha', 0.5)

    if t - 1 in temperatures:
        temp_base = temperatures[t - 1]
        k_base = k[t - 1]
    else:
        if temp_init is not None:
            temp = temp_init
        else:
            temp = scheme_acceptance_rate(**kwargs)
        # k controls reduction in error threshold
        k[t] = temp
        return temp
    
    if acceptance_rate < min_acceptance_rate:
        k_base = alpha * k_base

    k[t] = min(k_base, alpha * temp_base)
    temp = temp_base - k[t]
    
    return temp

