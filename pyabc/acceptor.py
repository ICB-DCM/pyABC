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
from typing import Callable, List


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
            target_acceptance_rate: float = 0.5,
            temp_max=None,
            temp_decay_exp: float = 3.0,
            use_target_acceptance_rate: bool = True,
            use_temp_decay_exp: bool = True):
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

        t_max: float, optional
        exponent: int, optional
        """

        super().__init__()

        self.pdf = pdf
        self.c = c
        self.target_acceptance_rate = target_acceptance_rate

        self.temp_max = temp_max
        self.temp_decay_exp = temp_decay_exp

        self.use_target_acceptance_rate = use_target_acceptance_rate
        self.use_temp_decay_exp = use_temp_decay_exp

        if not use_target_acceptance_rate and not use_temp_decay_exp:
            raise ValueError("No method to generate a temperature scheme "
                             "passed.")

        self.x_0 = None
        self.temperatures = {}
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

        # execute function
        initial_sum_stats = get_sum_stats()

        # update
        self._update(t, initial_sum_stats)

    def update(self,
               t: int,
               sum_stats: List[dict]):
        self._update(t, sum_stats)

    def _update(self,
                t: int,
                sum_stats: List[dict]):
        if self.c is None:
            self.c = self.pdf(self.x_0, self.x_0)

        values = [self.pdf(self.x_0, x) / self.c
                  for x in sum_stats]
        values = np.array(values)

        # compute optimal temperature for target acceptance rate
        if self.use_target_acceptance_rate or (not self.temperatures and self.temp_max is None:
            acceptance_rate_temp = self._compute_acceptance_rate_step(values)
        else:
            acceptance_rate_temp = np.inf

        # compute fall-back step according to decay scheme
        if self.use_temp_decay_exp:
            decay_temp = self._compute_decay_step(t, values)
        else:
            decay_temp = np.inf

        # take minimum
        temp = min(acceptance_rate_temp, decay_temp)

        # fill into temperatures list
        self.temperatures[t] = temp

    def _compute_acceptance_rate_step(self, values):

        # objective function which we wish to find a root for
        def obj(beta):
            val = np.sum(values**beta) / values.size - \
                self.target_acceptance_rate
            return val

        if obj(1) > 0:
            beta_opt = 1
        else:
            # perform binary search
            # TODO: take a more efficient optimization approach?
            beta_opt = sp.optimize.bisect(obj, 0, 1)

        # temperature is inverse beta
        temp_opt = 1 / beta_opt
        return temp_opt

    def _compute_decay_step(self, t, values):
        # check if we can compute a decay step
        if self.max_nr_populations == np.inf:
            # always take the acceptance rate step
            return np.inf

        # get temperature to start with
        if t - 1 in self.temperatures:
            temp_base = self.temperatures[t - 1]
        elif self.temp_max is not None:
            temp_base = self.temp_max
        else:
            # need to take the acceptance rate step
            return np.inf

        # how many steps left?
        t_to_go = self.max_nr_populations - (t - 1)
        if t_to_go < 2:
            # have to take exact step, i.e. a temperature of 1, next
            return 1.0
        temps = np.linspace(1, temp_base**(1 / self.temp_decay_exp),
                                t_to_go) ** self.temp_decay_exp

        temp = temps[-2]
        return temp

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
