import numpy as np
import scipy as sp
from typing import Callable, List

from .base import Distance


RET_SCALE_LIN = "RET_SCALE_LIN"
RET_SCALE_LOG = "RET_SCALE_LOG"
RET_SCALES = [RET_SCALE_LIN, RET_SCALE_LOG]


class StochasticKernel(Distance):
    """
    A stochastic kernel assesses the similarity between observed and
    simulated summary statistics or data via a probability measure.

    .. note::
        The returned value cannot be interpreted as a distance function,
        but rather as an inverse distance, as it increases as the similarity
        between observed and simulated summary statistics increases.
        Thus, a StochasticKernel should only be used together with a
        StochasticAcceptor.

    Parameters
    -----------

    ret_scale: str, optional (default = RET_SCALE_LIN)
        The scale of the value returned in __call__:
        Given a proability density p(x,x_0), the returned value
        can be either of p(x,x_0), or log(p(x,x_0)).

    keys: List[str]
        The keys of the summary statistics, specifying the order to be used.
    """

    def __init__(
            self,
            ret_scale=RET_SCALE_LIN,
            keys=None):
        StochasticKernel.check_ret_scale(ret_scale)
        self.ret_scale = ret_scale
        self.keys = keys

    def initialize(
            self,
            t: int,
            get_sum_stats: Callable[[], List[dict]],
            x_0: dict):
        """
        Remember the summary statistic keys in sorted order.
        """
        # initialize keys
        if self.keys is None:
            self.keys = sorted(x_0)

    @staticmethod
    def check_ret_scale(ret_scale):
        if ret_scale not in RET_SCALES:
            raise ValueError(
                f"ret_scale must be ofe of {RET_SCALES}")

    @property
    def pdf_max(self):
        """
        Return the maximum density function value possible under this
        noise model.

        Default: Return None.
        """
        return None

class SimpleFunctionKernel(StochasticKernel):
    """
    This is a wrapper around a simple function which calculates the
    probability density.

    Parameters
    ----------

    function: Callable[**kwargs, float]
        A Callable accepting a subset of __call__'s parameters.
        The function should be a pdf or pmf.

    ret_scale: as in StochasticKernel.ret_scale
    
    keys: List[str], optional (see StochasticKernel.keys)
    """
    def __init__(
            self,
            function,
            ret_scale=RET_SCALE_LIN,
            keys=None):
        super().__init__(ret_scale=ret_scale, keys=keys)
        self.function = function

    def __call__(
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        return self.function(x=x, x_0=x_0, t=t, par=par)


class NormalKernel(StochasticKernel):
    """
    A kernel with a normal, i.e. Gaussian,  probability density.
    This is just a wrapper around sp.multivariate_normal.

    Parameters
    ----------

    mean: array_like, optional (default = zeros vector)
        Mean of the distribution.

    cov: array_like, optional (default = identiy matrix)
        Covariance matrix of the distribution.

    ret_scale: str, optional (default = RET_SCALE_LIN)
        The scale on which the distribution is to be returned.

    keys: List[str], optional (see StochasticKernel.keys)

    .. note::

       The order of the entries in the mean and cov vectors is assumed
       to be the same as the one in keys. If keys is None, it is assumed to
       be the same as the one obtained via sorted(x.keys()) for summary
       statistics x.
    """

    def __init__(
            self,
            mean=None,
            cov=None,
            ret_scale=RET_SCALE_LIN,
            keys=None):
        super().__init__(ret_scale=ret_scale, keys=keys)
        self.mean = mean
        self.cov = cov

        # create frozen multivariate normal distribution
        self.rv = sp.stats.multivariate_normal(mean=self.mean, cov=self.cov)

        self.ret_scale = ret_scale
        self._pdf_max = None

    def initialize(
            self,
            t: int,
            get_sum_stats: Callable[[], List[dict]],
            x_0: dict):
        super().initialize(
            t=t,
            get_sum_stats=get_sum_stats,
            x_0=x_0)
        # cache pdf_max
        self._pdf_max = self(x_0, x_0)

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        """
        Return the value of the normal distribution at x - x_0, or its
        logarithm.
        """
        # difference to array
        diff = np.array([x[key] - x_0[key] for key in self.keys])
        
        # compute pdf
        if self.ret_scale == RET_SCALE_LIN:
            ret = self.rv.pdf(diff)
        else:  # self.ret_scale == RET_SCALE_LOG
            ret = self.rv.logpdf(diff)
        
        return ret

    @property
    def pdf_max(self):
        return self._pdf_max


class IndependentNormalKernel(StochasticKernel):
    """
    This kernel can be used for efficient computations of large-scale
    independent normal distributions, circumventing the covariance
    matrix, and performing computations directly on a log-scale to avoid
    numeric issues.

    Parameters
    ----------

    mean: array_like, optional (default = zeros vector)
        Mean of the distribution.

    var: array_like, optional (default = ones vector)
        Variances of the distribution (assuming zeros in the off-diagonal
        of the covariance matrix).

    keys: List[str], optional (see StochasticKernel.keys)
    """

    def __init__(
            self,
            mean=None,
            var=None,
            keys=None):
        super().__init__(ret_scale=RET_SCALE_LOG, keys=keys)
        self.mean = None
        self.var = None
        self.dim = None

    def initialize(
            self,
            t: int,
            get_sum_stats: Callable[[], List[dict]],
            x_0: dict):
        super().initialize(
            t=t,
            get_sum_stats=get_sum_stats,
            x_0=x_0)
        
        # set dimension
        self.dim = len(x_0)

        # initialize mean correctly
        if self.mean is None:
            self.mean = np.zeros(self.dim)
        else:
            self.mean = np.array(self.mean) * np.ones(self.dim)

        # initialize var correctly
        if self.var is None:
            self.var = np.ones(self.dim)
        else:
            self.var = np.array(self.var) * np.ones(self.dim)

        # cache pdf_max
        self._pdf_max = self(x_0, x_0)

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None):
        # difference to array
        diff = np.array([x[key] - x_0[key] for key in self.keys])

        # compute pdf
        log_2_pi = np.sum(np.log(2 * np.pi * self.var))

        squares = np.sum((diff**2) / self.var)

        log_pdf = - 0.5 * (log_2_pi + squares)

        return log_pdf

    @property
    def pdf_max(self):
        return self._pdf_max


class BinomialKernel(StochasticKernel):
    """
    A kernel with a binomial probability mass function.

    Parameters
    ----------

    p: float
        The success probability.

    ret_scale: str, optional (default = RET_SCALE_LIN)
        The scale on which the distribution is to be returned.

    keys: List[str], optional (see StochasticKernel.keys)
    """

    def __init__(
            self,
            p: float,
            ret_scale=RET_SCALE_LIN,
            keys=None):
        super().__init__(ret_scale=ret_scale, keys=keys)
        
        if p > 1 or p < 0:
            raise ValueError(
                f"p must be in the interval [0, 1], but obtained {p}.")
        self.p = p

    def __call__(
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        x = np.array([x[key] for key in self.keys])
        x_0 = np.array([x_0[key] for key in self.keys])

        if self.ret_scale == RET_SCALE_LIN:
            ret = 1.0
            for j in range(len(x_0)):
                ret *= sp.stats.binom.pmf(k=x_0[j], n=x[j], p=self.p) \
                       if x[j] > 0 else 1
        else:  # self.ret_scale == RET_SCALE_LOG
            ret = 0.0
            for j in range(len(x_0)):
                ret += sp.stats.binom.logpmf(k=x_0[j], n=x[j], p=self.p) \
                       if x[j] > 0 else 0

        return ret

    @property
    def pdf_max(self):
        """
        This value is most certaily much too high in practice.
        """
        return 1
