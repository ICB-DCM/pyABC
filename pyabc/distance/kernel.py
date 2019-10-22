import numpy as np
import scipy as sp
from typing import Callable, List

from .base import Distance


SCALE_LIN = "SCALE_LIN"
SCALE_LOG = "SCALE_LOG"
SCALES = [SCALE_LIN, SCALE_LOG]


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

    ret_scale: str, optional (default = SCALE_LIN)
        The scale of the value returned in __call__:
        Given a proability density p(x,x_0), the returned value
        can be either of p(x,x_0), or log(p(x,x_0)).

    keys: List[str], optional
        The keys of the summary statistics, specifying the order to be used.

    pdf_max: float, optional
        The maximum possible probability density function value.
        Defaults to None and is then computed as the density at (x_0, x_0),
        where x_0 denotes the observed summary statistics.
        Must be overridden if pdf_max is to be used in the analysis by the
        acceptor.
        This value should be in the scale specified by ret_scale already.
    """

    def __init__(
            self,
            ret_scale=SCALE_LIN,
            keys=None,
            pdf_max=None):
        StochasticKernel.check_ret_scale(ret_scale)
        self.ret_scale = ret_scale
        self.keys = keys
        self.pdf_max = pdf_max

    def initialize(
            self,
            t: int,
            get_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        """
        Remember the summary statistic keys in sorted order,
        if not set in __init__ already.
        """
        # initialize keys
        if self.keys is None:
            self.initialize_keys(x_0)

    @staticmethod
    def check_ret_scale(ret_scale):
        if ret_scale not in SCALES:
            raise ValueError(
                f"The ret_scale {ret_scale} must be one of {SCALES}.")

    def initialize_keys(self, x):
        self.keys = sorted(x)


class SimpleFunctionKernel(StochasticKernel):
    """
    This is a wrapper around a simple function which calculates the
    probability density.

    Parameters
    ----------

    fun: Callable[**kwargs, float]
        A Callable accepting a subset of __call__'s parameters.
        The function should be a pdf or pmf.

    ret_scale, keys, pdf_max: as in StochasticKernel
    """

    def __init__(
            self,
            fun,
            ret_scale=SCALE_LIN,
            keys=None,
            pdf_max=None):
        super().__init__(ret_scale=ret_scale, keys=keys, pdf_max=pdf_max)
        self.fun = fun

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        return self.fun(x=x, x_0=x_0, t=t, par=par)


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

    ret_scale, keys, pdf_max: As in StochasticKernel.


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
            ret_scale=SCALE_LIN,
            keys=None,
            pdf_max=None):
        super().__init__(ret_scale=ret_scale, keys=keys, pdf_max=pdf_max)
        self.mean = mean
        self.cov = cov

        # create frozen multivariate normal distribution
        self.rv = sp.stats.multivariate_normal(mean=self.mean, cov=self.cov)

        self.ret_scale = ret_scale

    def initialize(
            self,
            t: int,
            get_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        # in particular set keys
        super().initialize(
            t=t,
            get_sum_stats=get_sum_stats,
            x_0=x_0)

        # cache pdf_max
        if self.pdf_max is None:
            # take  value at observed summary statistics
            self.pdf_max = self(x_0, x_0)

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
        if self.keys is None:
            self.initialize_keys(x_0)

        # difference to array
        diff = np.array([x[key] - x_0[key] for key in self.keys])

        # compute pdf
        if self.ret_scale == SCALE_LIN:
            ret = self.rv.pdf(diff)
        else:  # self.ret_scale == SCALE_LOG
            ret = self.rv.logpdf(diff)

        return ret


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

    var: Union[array_like, Callable], optional (default = ones vector)
        Variances of the distribution (assuming zeros in the off-diagonal
        of the covariance matrix). Can also be a Callable taking as
        arguments the parameters. In that case, pdf_max should also be given
        if it is supposed to be used. Usually, it will then be given as the
        density at the observed statistics assuming the minimum allowed
        variance.

    keys, pdf_max: As in StochasticKernel.

    """

    def __init__(
            self,
            mean=None,
            var=None,
            keys=None,
            pdf_max=None):
        super().__init__(ret_scale=SCALE_LOG, keys=keys, pdf_max=pdf_max)
        self.mean = mean
        self.var = var
        self.dim = None

    def initialize(
            self,
            t: int,
            get_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        # in particular set keys
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
        if not callable(self.var):
            self.var = np.array(self.var) * np.ones(self.dim)

        # cache pdf_max (from now on __call__ can be used)
        if self.pdf_max is None:
            # take value at observed summary statistics
            self.pdf_max = self(x_0, x_0)

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None):
        if self.keys is None:
            self.initialize_keys(x_0)

        # compute variance
        if callable(self.var):
            # parameterized variance (i.e. probably estimated)
            var = self.var(par)
        else:
            # constant numeric values
            var = self.var

        # difference to array
        diff = np.array([x[key] - x_0[key] for key in self.keys])

        # compute pdf
        log_2_pi = np.sum(np.log(2 * np.pi * var))

        squares = np.sum((diff**2) / var)

        log_pdf = - 0.5 * (log_2_pi + squares)

        return log_pdf


class IndependentLaplaceKernel(StochasticKernel):
    """
    This kernel can be used for efficient computations of large-scale
    independent Laplace distributions, performing computations directly
    on a log-scale to avoid numeric issues. In each coordinate, a 1-dim
    Laplace distribution

    .. math::
        p(x) = \\frac{1}{2b}\\exp (\\frac{1}{b}|x-a|)

    is assumed.

    Parameters
    ----------

    mean: array_like, optional (default = zeros vector)
        Mean of the distribution.

    scale: Union[array_like, Callable], optional (default = ones vector)
        Scale terms b of the distribution. Can also be a Callable taking as
        arguments the parameters. In that case, pdf_max should also be given
        if it is supposed to be used. Usually, it will then be given as the
        density at the observed statistics assuming the minimum allowed
        variance.

    keys, pdf_max: As in StochasticKernel.

    """

    def __init__(
            self,
            mean=None,
            scale=None,
            keys=None,
            pdf_max=None):
        super().__init__(ret_scale=SCALE_LOG, keys=keys, pdf_max=pdf_max)
        self.mean = mean
        self.scale = scale
        self.dim = None

    def initialize(
            self,
            t: int,
            get_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        # in particular set keys
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
        if self.scale is None:
            self.scale = np.ones(self.dim)
        if not callable(self.scale):
            self.scale = np.array(self.scale) * np.ones(self.dim)

        # cache pdf_max (from now on __call__ can be used)
        if self.pdf_max is None:
            # take value at observed summary statistics
            self.pdf_max = self(x_0, x_0)

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None):
        if self.keys is None:
            self.initialize_keys(x_0)

        # compute variance
        if callable(self.scale):
            # parameterized variance (i.e. probably estimated)
            scale = self.scale(par)
        else:
            # constant numeric values
            scale = self.scale

        # difference to array
        diff = np.array([x[key] - x_0[key] for key in self.keys])

        # compute pdf
        log_2_b = np.sum(np.log(2 * scale))

        abs_diff = np.sum(np.abs(diff) / scale)

        log_pdf = - (log_2_b + abs_diff)

        return log_pdf


class BinomialKernel(StochasticKernel):
    """
    A kernel with a binomial probability mass function.

    Parameters
    ----------

    p: float
        The success probability.

    ret_scale: str, optional (default = SCALE_LIN)
        The scale on which the distribution is to be returned.

    keys: List[str], optional (see StochasticKernel.keys)

    pdf_max: float, optional (see StochasticKernel.pdf_max)
    """

    def __init__(
            self,
            p: float,
            ret_scale=SCALE_LIN,
            keys=None,
            pdf_max=None):
        super().__init__(ret_scale=ret_scale, keys=keys, pdf_max=pdf_max)

        if p > 1 or p < 0:
            raise ValueError(
                f"The success probability p={p} must be in the interval"
                f"[0, 1].")
        self.p = p

    def initialize(
            self,
            t: int,
            get_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        # in particular set keys
        super().initialize(
            t=t,
            get_sum_stats=get_sum_stats,
            x_0=x_0)

        # cache pdf_max (from now on __call__ can be used)
        if self.pdf_max is None:
            # take value at observed summary statistics
            self.pdf_max = binomial_pdf_max(
                x_0, self.keys, self.p, self.ret_scale)

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        x = np.array([x[key] for key in self.keys], dtype=int).flatten()
        x_0 = np.array([x_0[key] for key in self.keys], dtype=int).flatten()
        p = self.p

        if self.ret_scale == SCALE_LIN:
            print(sp.stats.binom.pmf(k=x_0, n=x, p=p))
            ret = np.prod(sp.stats.binom.pmf(k=x_0, n=x, p=p))
        else:  # self.ret_scale == SCALE_LOG
            ret = np.sum(sp.stats.binom.logpmf(k=x_0, n=x, p=p))

        return ret


def binomial_pdf_max(x_0, keys, p, ret_scale):
    ks = np.array([x_0[key] for key in keys], dtype=int).flatten()
    ns = np.maximum(np.floor((ks - p) / p), 0)
    pms = sp.stats.binom.logpmf(k=ks, n=ns, p=p)
    log_pdf_max = np.sum(pms)

    if ret_scale == SCALE_LIN:
        return np.eps(log_pdf_max)
    return log_pdf_max
