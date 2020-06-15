"""Stochastic kernels."""

import numpy as np
from scipy import stats
from typing import Callable, List, Union

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
        acceptor and the default is not applicable.
        This value should be in the scale specified by ret_scale already.
    """

    def __init__(
            self,
            ret_scale: str = SCALE_LIN,
            keys: List[str] = None,
            pdf_max: float = None):
        StochasticKernel.check_ret_scale(ret_scale)
        self.ret_scale = ret_scale
        self.keys = keys
        self.pdf_max = pdf_max

    def initialize(
            self,
            t: int,
            get_all_sum_stats: Callable[[], List[dict]],
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

    fun: Callable
        A Callable accepting `__call__`'s parameters.
        The function should be a pdf or pmf.
    ret_scale, keys, pdf_max: as in StochasticKernel
    """

    def __init__(
            self,
            fun: Callable,
            ret_scale: str = SCALE_LIN,
            keys: List[str] = None,
            pdf_max: float = None):
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
            cov: np.ndarray = None,
            ret_scale: str = SCALE_LOG,
            keys: List[str] = None,
            pdf_max: float = None):
        super().__init__(ret_scale=ret_scale, keys=keys, pdf_max=pdf_max)
        self.cov = cov
        self.ret_scale = ret_scale

    def initialize(
            self,
            t: int,
            get_all_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        # in particular set keys
        super().initialize(
            t=t,
            get_all_sum_stats=get_all_sum_stats,
            x_0=x_0)

        # initialize distribution
        self._init_distr(x_0)

        # cache pdf_max
        if self.pdf_max is None:
            # take  value at observed summary statistics
            self.pdf_max = self(x_0, x_0)

    def _init_distr(self, x_0):
        """
        Initialize cov (covariance) and rv (distribution).
        """
        if self.cov is None:
            dim = sum(np.size(x_0[key]) for key in self.keys)
            self.cov = np.eye(dim)
        self.cov = np.asarray(self.cov)
        # create frozen multivariate normal distribution
        dim = self.cov.shape[0]
        mean = np.zeros(dim)
        self.rv = stats.multivariate_normal(mean=mean, cov=self.cov)

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
        # safety check
        if self.keys is None:
            self.initialize_keys(x_0)

        # difference to array
        diff = _diff_arr(x, x_0, self.keys)

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

    var: Union[array_like, float, Callable], optional (default = ones vector)
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
            var: Union[Callable, List[float], float] = None,
            keys: List[str] = None,
            pdf_max: float = None):
        super().__init__(ret_scale=SCALE_LOG, keys=keys, pdf_max=pdf_max)
        self.var = var

    def initialize(
            self,
            t: int,
            get_all_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        # in particular set keys
        super().initialize(
            t=t,
            get_all_sum_stats=get_all_sum_stats,
            x_0=x_0)

        # dimension
        dim = sum(np.size(x_0[key]) for key in self.keys)

        # initialize var
        if self.var is None:
            self.var = np.ones(dim)
        if not callable(self.var):
            self.var = np.asarray(self.var) * np.ones(dim)

        # cache pdf_max
        if self.pdf_max is None and not callable(self.var):
            # take value at observed summary statistics
            self.pdf_max = self(x_0, x_0)

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None):
        # safety check
        if self.keys is None:
            self.initialize_keys(x_0)

        # compute variance
        var = self.var(par) if callable(self.var) else self.var
        var = np.asarray(var)

        # difference to array
        diff = _diff_arr(x, x_0, self.keys)

        if var.size == 1:
            var = var * np.ones(diff.size)

        # compute pdf
        log_2_pi = np.sum(np.log(2) + np.log(np.pi) + np.log(var))

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

    scale: Union[array_like, float, Callable], optional (default = ones vector)
        Scale terms b of the distribution. Can also be a Callable taking as
        arguments the parameters. In that case, pdf_max should also be given
        if it is supposed to be used. Usually, it will then be given as the
        density at the observed statistics assuming the minimum allowed
        variance.

    keys, pdf_max: As in StochasticKernel.

    """

    def __init__(
            self,
            scale: Union[Callable, List[float], float] = None,
            keys: List[str] = None,
            pdf_max: float = None):
        super().__init__(ret_scale=SCALE_LOG, keys=keys, pdf_max=pdf_max)
        self.scale = scale
        self.dim = None

    def initialize(
            self,
            t: int,
            get_all_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        # in particular set keys
        super().initialize(
            t=t,
            get_all_sum_stats=get_all_sum_stats,
            x_0=x_0)

        # dimension
        dim = sum(np.size(x_0[key]) for key in self.keys)

        # initialize scale correctly
        if self.scale is None:
            self.scale = np.ones(dim)
        if not callable(self.scale):
            self.scale = np.asarray(self.scale) * np.ones(dim)

        # cache pdf_max
        if self.pdf_max is None and not callable(self.scale):
            # take value at observed summary statistics
            self.pdf_max = self(x_0, x_0)

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None):
        # safety check
        if self.keys is None:
            self.initialize_keys(x_0)

        # compute variance
        scale = self.scale(par) if callable(self.scale) else self.scale
        scale = np.asarray(scale)

        # difference to array
        diff = _diff_arr(x, x_0, self.keys)

        if scale.size == 1:
            scale = scale * np.ones(diff.size)

        # compute pdf
        log_2_b = np.sum(np.log(2) + np.log(scale))

        abs_diff = np.sum(np.abs(diff) / scale)

        log_pdf = - (log_2_b + abs_diff)

        return log_pdf


class BinomialKernel(StochasticKernel):
    """
    A kernel with a binomial probability mass function.

    Parameters
    ----------

    p: Union[float, Callable]
        The success probability.
    ret_scale, keys, pdf_max: See StochasticKernel.
    """

    def __init__(
            self,
            p: Union[float, Callable],
            ret_scale: str = SCALE_LOG,
            keys: List[str] = None,
            pdf_max: float = None):
        super().__init__(ret_scale=ret_scale, keys=keys, pdf_max=pdf_max)

        if not callable(p) and (p > 1 or p < 0):
            raise ValueError(
                f"The success probability p={p} must be in the interval"
                f"[0, 1].")
        self.p = p

    def initialize(
            self,
            t: int,
            get_all_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        # in particular set keys
        super().initialize(
            t=t,
            get_all_sum_stats=get_all_sum_stats,
            x_0=x_0)

        # cache pdf_max
        if self.pdf_max is None and not callable(self.p):
            # take value at observed summary statistics
            self.pdf_max = binomial_pdf_max(
                x_0, self.keys, self.p, self.ret_scale)

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        x = np.asarray(_arr(x, self.keys), dtype=int)
        x_0 = np.asarray(_arr(x_0, self.keys), dtype=int)

        # compute p
        p = self.p if not callable(self.p) else self.p(par)

        if self.ret_scale == SCALE_LIN:
            ret = np.prod(stats.binom.pmf(k=x_0, n=x, p=p))
        else:  # self.ret_scale == SCALE_LOG
            ret = np.sum(stats.binom.logpmf(k=x_0, n=x, p=p))

        return float(ret)


class PoissonKernel(StochasticKernel):
    """
    A kernel with a Poisson probability mass function.

    Parameters
    ----------
    ret_scale, keys, pdf_max: See StochasticKernel.
    """

    def __init__(
            self,
            ret_scale: str = SCALE_LOG,
            keys: List[str] = None,
            pdf_max: float = None):
        super().__init__(ret_scale=ret_scale, keys=keys, pdf_max=pdf_max)

    def initialize(
            self,
            t: int,
            get_all_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        # in particular set keys
        super().initialize(
            t=t,
            get_all_sum_stats=get_all_sum_stats,
            x_0=x_0)

        # cache pdf_max
        if self.pdf_max is None:
            # take value at observed summary statistics
            # this is the optimal value
            self.pdf_max = self(x_0, x_0)

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        x = np.asarray(_arr(x, self.keys), dtype=int)
        x_0 = np.asarray(_arr(x_0, self.keys), dtype=int)

        if self.ret_scale == SCALE_LIN:
            ret = np.prod(stats.poisson.pmf(k=x_0, mu=x))
        else:  # self.ret_scale == SCALE_LOG
            ret = np.sum(stats.poisson.logpmf(k=x_0, mu=x))

        return float(ret)


class NegativeBinomialKernel(StochasticKernel):
    """
    A kernel with a negative binomial probability mass function.

    Parameters
    ----------

    p: Union[float, Callable]
        The success probability.
    ret_scale, keys, pdf_max: See StochasticKernel.
    """

    def __init__(
            self,
            p: float,
            ret_scale: str = SCALE_LOG,
            keys: List[str] = None,
            pdf_max: float = None):
        super().__init__(ret_scale=ret_scale, keys=keys, pdf_max=pdf_max)

        if not callable(p) and (p > 1 or p < 0):
            raise ValueError(
                f"The success probability p={p} must be in the interval"
                f"[0, 1].")
        self.p = p

    def initialize(
            self,
            t: int,
            get_all_sum_stats: Callable[[], List[dict]],
            x_0: dict = None):
        # in particular set keys
        super().initialize(
            t=t,
            get_all_sum_stats=get_all_sum_stats,
            x_0=x_0)

        # pdf_max is not computed

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
        x = np.asarray(_arr(x, self.keys), dtype=int)
        x_0 = np.asarray(_arr(x_0, self.keys), dtype=int)

        # compute p
        p = self.p if not callable(self.p) else self.p(par)

        if self.ret_scale == SCALE_LIN:
            ret = np.prod(stats.nbinom.pmf(k=x_0, n=x, p=p))
        else:  # self.ret_scale == SCALE_LOG
            ret = np.sum(stats.nbinom.logpmf(k=x_0, n=x, p=p))

        return float(ret)


def binomial_pdf_max(x_0, keys, p, ret_scale):
    """
    Compute the model value of the binomial distribution.

    Note that since we interpret x_0 as the noisy k value, we search
    the model value over arbitrary n.

    The optimal value was calculated by checking p(n+1,k) / p(n,k).
    """
    ks = np.asarray(_arr(x_0, keys), dtype=int)
    ns = np.maximum(np.ceil((ks - p) / p), 0)
    pms = stats.binom.logpmf(k=ks, n=ns, p=p)

    # sum over all log values
    log_pdf_max = np.sum(pms)

    if ret_scale == SCALE_LIN:
        return np.exp(log_pdf_max)
    return log_pdf_max


def _diff_arr(x, x_0, keys):
    """
    Get difference array.
    """
    diff = []
    for key in keys:
        d = x[key] - x_0[key]
        try:
            diff.extend(d)
        except Exception:
            diff.append(d)
    diff = np.asarray(diff)
    return diff


def _arr(x, keys):
    """
    Get as flat array.
    """
    arr = []
    for key in keys:
        val = x[key]
        try:
            arr.extend(val)
        except Exception:
            arr.append(val)
    arr = np.asarray(arr)
    return arr
