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

import numpy as np
import pandas as pd
from typing import Callable
import logging

from ..distance import Distance, SCALE_LIN, StochasticKernel
from ..epsilon import Epsilon
from ..parameters import Parameter
from .pdf_norm import pdf_norm_max_found
from ..storage import save_dict_to_json


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
        distance_function: Distance
            Distance object. The acceptor should not modify it, but might
            extract some meta information.
        x_0: dict
            The observed summary statistics.
        """
        pass

    def update(self,
               t: int,
               get_weighted_distances: Callable[[], pd.DataFrame],
               prev_temp: float,
               acceptance_rate: float):
        """
        Update the acceptance criterion.

        Parameters
        ----------

        t: int
            The timepoint to initialize the acceptor for.
        get_weighted_distances: Callable[[], pd.DataFrame]
            The past generation's weighted distances.
        prev_temp: float
            The past generation's temperature.
        acceptance_rate: float
            The past generation's acceptance rate.
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

    # pylint: disable=R0201
    def get_epsilon_config(self, t: int) -> dict:
        """
        Create a configuration object that contains all values of interest for
        the update of the Epsilon object.

        Parameters
        ----------
        t: int
            The timepoint for which to get the config.

        Returns
        -------
        config: dict
            The relevant information.
        """
        return None


class SimpleFunctionAcceptor(Acceptor):
    """
    Initialize from function.

    Parameters
    ----------

    fun: Callable, optional
        Callable with the same signature as the __call__ method.
    """

    def __init__(self, fun: Callable):
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


class StochasticAcceptor(Acceptor):
    """
    This acceptor implements a stochastic acceptance step based on a
    probability density, generalizing from the uniform acceptance kernel.
    A particle is accepted if for the simulated summary statistics x,
    observed summary statistics x_0 and parameters theta holds

    .. math::

       \\frac{\\text{pdf}(x_0|x,\\theta)}{c}\\geq u

    where u ~ U[0,1], and c is a normalizing constant.

    The concept is based on [#wilkinson]_. In addition, we introduce
    acceptance kernel temperation and rejection control importance sampling
    to permit a more flexible choice and adaptation of c.

    .. [#wilkinson] Wilkinson, Richard David; "Approximate Bayesian
        computation (ABC) gives exact results under the assumption of model
        error"; Statistical applications in genetics and molecular biology
        12.2 (2013): 129-141.

    """

    def __init__(
            self,
            pdf_norm_method: Callable = None,
            apply_importance_weighting: bool = True,
            log_file: str = None):
        """
        Parameters
        ----------
        pdf_norm_method: Callable, optional
            Function to calculate a pdf normalization (denoted `c` above).
            Shipped are `pyabc.acceptor.pdf_norm_from_kernel` to use the
            value provided by the StochasticKernel, and
            `pyabc.acceptor.pdf_norm_max_found` (default) to always use
            the maximum value among accepted particles so far.
            Note that re-weighting based on ideas from rejection control
            importance sampling to handle the normalization constant being
            insufficient, and thus avoiding an importance sampling bias,
            is included either way.
        apply_importance_weighting: bool, optional
            Whether to apply weights to correct for a bias induced by
            samples exceeding the density normalization. This may be False
            usually only for testing purposes.
        log_file: str, optional
            A log file for storing data of the acceptor that are currently not
            saved in the database. The data are saved in json format and can
            be retrieved via `pyabc.storage.load_dict_from_json`.
        """
        super().__init__()

        if pdf_norm_method is None:
            pdf_norm_method = pdf_norm_max_found
        self.pdf_norm_method = pdf_norm_method
        self.apply_importance_weighting = apply_importance_weighting
        self.log_file = log_file

        # maximum pdfs, indexed by time
        self.pdf_norms = {}

        # fields to be filled later
        self.x_0 = None
        self.kernel_scale = None
        self.kernel_pdf_max = None

    def initialize(
            self,
            t: int,
            get_weighted_distances: Callable[[], pd.DataFrame],
            distance_function: StochasticKernel,
            x_0: dict):
        """
        Initialize temperature and maximum pdf.
        """
        self.x_0 = x_0
        self.kernel_scale = distance_function.ret_scale
        self.kernel_pdf_max = distance_function.pdf_max

        # update
        self._update(t, get_weighted_distances)

    def update(self,
               t: int,
               get_weighted_distances: Callable[[], pd.DataFrame],
               prev_temp: float,
               acceptance_rate: float):
        self._update(t, get_weighted_distances, prev_temp, acceptance_rate)

    def _update(self,
                t: int,
                get_weighted_distances: Callable[[], pd.DataFrame],
                prev_temp: float = None,
                acceptance_rate: float = 1.0):
        """
        Update schemes for the upcoming time point t.
        """
        # update pdf normalization
        pdf_norm = self.pdf_norm_method(
            kernel_val=self.kernel_pdf_max,
            get_weighted_distances=get_weighted_distances,
            prev_pdf_norm=None if not self.pdf_norms
            else max(self.pdf_norms.values()),
            acceptance_rate=acceptance_rate,
            prev_temp=prev_temp)
        self.pdf_norms[t] = pdf_norm

        self.log(t)

    def log(self, t):
        logger.debug(f"pdf_norm={self.pdf_norms[t]:.4e} for t={t}.")

        if self.log_file:
            save_dict_to_json(self.pdf_norms, self.log_file)

    def get_epsilon_config(self, t: int) -> dict:
        """
        Pack the pdf normalization and the kernel scale.
        """
        return dict(
            pdf_norm=self.pdf_norms[t],
            kernel_scale=self.kernel_scale,  # TODO Refactor
        )

    def __call__(self,
                 distance_function: StochasticKernel,
                 eps, x, x_0, t, par):
        # rename
        kernel = distance_function

        # temperature
        temp = eps(t)

        # compute probability density
        density = kernel(x, x_0, t, par)

        pdf_norm = self.pdf_norms[t]

        # compute acceptance probability
        if kernel.ret_scale == SCALE_LIN:
            acc_prob = (density / pdf_norm) ** (1 / temp)
        else:  # kernel.ret_scale == SCALE_LOG
            acc_prob = np.exp((density - pdf_norm) * (1 / temp))

        # accept
        threshold = np.random.uniform(low=0, high=1)
        if acc_prob >= threshold:
            accept = True
        else:
            accept = False

        # weight
        if acc_prob == 0.0:
            weight = 0.0
        elif self.apply_importance_weighting:
            weight = acc_prob / min(1, acc_prob)
        else:
            weight = 1.0

        # check pdf max ok
        if pdf_norm < density:
            logger.debug(
                f"Encountered density={density:.4e} > c={pdf_norm:.4e}, "
                f"thus weight={weight:.4e}.")

        # return unscaled density value and the acceptance flag
        return AcceptorResult(density, accept, weight)
