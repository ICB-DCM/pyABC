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
import scipy.stats as sp_stats


class Acceptor:
    """
    This class encodes the acceptance step.
    """

    def __init__(self):
        """
        Default constructor.
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

       \\frac{pdf(x_0|x)}{c}\geq u

    where u ~ U[0,1], and c is a normalizing constant.

    The implementation is based on [#wilkinson].

    .. [#wilkinson] Wilkinson, Richard David; "Approximate Bayesian
       computation (ABC) gives exact results under the assumption of model
       error"; Statistical applications in genetics and molecular biology
       12.2 (2013): 129-141.
    """

    def __init__(
            self,
            pdf=None,
            c=None,
            t_max=None,
            exponent=3):
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

        if t_max is None:
            t_max = 42
        self.t_max = t_max
        self.exponent = exponent

    def __call__(self, t, distance_function, eps, x, x_0, pars):
        if t >= self.nr_populations:
            raise ValueError("PFUI!")
        temps = np.linspace(1,
                            self.max_temp**(1 / self.exp),
                            self.nr_populations) ** self.exp
        temp = temps[self.nr_populations - 1 - t]
        beta = 1 / temp

        # extract summary statistics as array
        x = np.asarray(list(x.values()))
        x_0 = np.asarray(list(x_0.values()))
        n = len(x)

        # noise distribution
        if self.distribution is None:
            distribution = sp_stats.multivariate_normal(
                mean=np.zeros(n), cov=np.eye(n))
        else:
            distribution = self.distribution

        # maximum probability density
        if self.max_density is None:
            max_density = distribution.pdf(np.zeros(n))
        else:
            max_density = self.max_density

        # compute probability density
        density = distribution.pdf(x - x_0)

        # acceptance probability
        acceptance_probability = density**beta / max_density**beta

        # accept
        threshold = np.random.uniform(low=0, high=1)
        if acceptance_probability >= threshold:
            accept = True
            # print(acceptance_probability, threshold)
        else:
            accept = False

        return np.nan, accept
