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


class Acceptor:
    """
    This class encodes the acceptance step.
    """

    def __init__(self):
        """
        Default constructor.
        """
        pass

    def __call__(self, t, distance_function, eps, x, x_0):
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
        Callable with the same signature as the __call__ method. Per default,
        accept_use_current_time is used.
    """
    def __init__(self, fun=None):
        super().__init__()

        if fun is None:
            fun = accept_use_current_time
        self.fun = fun

    def __call__(self, t, distance_function, eps, x, x_0):
        return self.fun(t, distance_function, eps, x, x_0)

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


def accept_use_current_time(t, distance_function, eps, x, x_0):
    """
    Use only the distance function and epsilon criterion at the current time
    point to evaluate whether to accept or reject.
    """

    d = distance_function(t, x, x_0)
    accept = d <= eps(t)

    return d, accept


def accept_use_complete_history(t, distance_function, eps, x, x_0):
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
