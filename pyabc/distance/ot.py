"""Optimal transport distances."""

import logging
from functools import partial
from typing import Callable, Union

import numpy as np
import scipy.linalg as la
import scipy.spatial as spat

from ..population import Sample
from ..sumstat import Sumstat
from .base import Distance

try:
    import ot
except ImportError:
    ot = None


logger = logging.getLogger("ABC.Distance")


class WassersteinDistance(Distance):
    r"""Optimal transport Wasserstein distance between empirical distributions.

    The Wasserstein distance, also referred to as Vaserstein,
    Kantorovich-Rubinstein, or earth mover's distance,
    is a metric between probability distributions on a given metric space
    (M, d).
    Intuitively, it quantifies the minimum cost of transforming one probability
    distribution on M into another, with point-wise cost function d.

    The Wasserstein distance between discrete distributions
    :math:`\mu = \{(x_i,a_i)\}` and :math:`\nu = \{(y_i,b_i)\}` can be
    expressed as

    .. math::
        W_p(\mu,\nu) = \left(\sum_{i,j}\gamma^*_{ij}M_{ij}\right)^{1/p}

    where the optimal transport mapping is given as

    .. math::
        \gamma^* = \text{argmin}_{\gamma \in \mathbb{R}^{m\times n}}
        \sum_{i,j}\gamma_{ij}M_{ij}

        s.t. \gamma 1 = a; \gamma^T 1= b; \gamma\geq 0

    where :math:`M\in\mathbb{R}^{m\times n}` is the pairwise cost matrix
    defining the cost to move mass from bin :math:`x_i`
    to bin :math:`y_j`,
    e.g. expressed via a distance metric, :math:`M_{ij} = \|x_i - y_j\|_p`,
    and :math:`a` and :math:`b` are histograms weighting samples
    (e.g. uniform).

    Its application in ABC is based on [#bernton2019]_.
    For further information see e.g.
    https://en.wikipedia.org/wiki/Wasserstein_metric.

    .. [#bernton2019]
        Bernton, E., Jacob, P.E., Gerber, M. and Robert, C.P.,
        2019.
        Approximate Bayesian computation with the Wasserstein distance.
        arXiv preprint arXiv:1905.03747.
    """

    def __init__(
        self,
        sumstat: Sumstat,
        p: float = 2.0,
        dist: Union[str, Callable] = None,
        emd_args: dict = None,
    ):
        """
        Parameters
        ----------
        sumstat:
            Summary statistics. Returns a ndarray of shape (n, dim), where
            n is the number of samples and dim the sample dimension.
        p:
            Distance exponent, e.g. Manhattan (p=1), Euclidean (p=2).
            If dist is separately specified, ^(1/p) is still applied at the
            end.
        dist:
            Distance to use. If not specified, the distance is induced by p.
        emd_args:
            Further keyword arguments passed on to ot.emd.
        """
        if ot is None:
            raise ImportError(
                "This distance requires the optimal transport library pot. "
                "Install via `pip install pyabc[ot]` or `pip install pot`.",
            )
        super().__init__()

        self.sumstat: Sumstat = sumstat
        self.p: float = p

        # distance function
        if dist is None:
            # translate from p
            if p == 1.0:
                dist = "cityblock"
            elif p == 2.0:
                dist = "sqeuclidean"
            else:
                # of course, we could permit arbitrary norms here if needed
                raise ValueError(f"Cannot translate p={p} into a distance.")
        if isinstance(dist, str):
            dist = partial(spat.distance.cdist, metric=dist)
        self.dist: Callable = dist

        if emd_args is None:
            emd_args = {}
        self.emd_args: dict = emd_args

        # observed data
        self.x0: Union[dict, None] = None
        self.s0: Union[np.ndarray, None] = None

    def initialize(
        self,
        x_0: dict,
        t: int = None,
        get_sample: Callable[[], Sample] = None,
        total_sims: int = None,
    ) -> None:
        # initialize summary statistics
        self.sumstat.initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )

        # observed data
        self.x0 = x_0
        self.s0 = self.sumstat(self.x0)

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        total_sims: int,
    ) -> bool:
        # update summary statistics
        updated = self.sumstat.update(
            t=t,
            get_sample=get_sample,
            total_sims=total_sims,
        )
        if updated:
            self.s0 = self.sumstat(self.x0)
        return updated

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None,
    ) -> float:
        # compute summary statistics, shape (n, dim), (n0, dim)
        s, s0 = self.sumstat(x), self.sumstat(x_0)
        n, n0 = s.shape[0], s0.shape[0]

        # pairwise cost matrix, shape (n, n0)
        m = self.dist(XA=s, XB=s0)

        # weights (could also be passed/learned?)
        w, w0 = np.ones((n,)) / n, np.ones((n0,)) / n0

        # optimal transport ("earth mover's") cost value
        cost = ot.emd2(a=w, b=w0, M=m, **self.emd_args, log=False)

        # take root to match Wasserstein distance definition
        if self.p < np.inf:
            cost = cost ** (1 / self.p)

        return cost


class SlicedWassersteinDistance(Distance):
    r"""Sliced Wasserstein distance via efficient one-dimensional projections.

    As the optimal transport mapping underlying Wasserstein distances can be
    challenging for high-dimensional problems, this distance reduces
    multi-dimensional distributions to one-dimensional representations via
    linear projections, and then averages 1d Wasserstein distances, which
    can be efficiently calculated by sorting,
    across the projected distributions.

    More explicitly, with
    :math:`\mathbb{S}^{d-1} = \{u\in\mathbb{R}^d: \|x\|_2=1\}` denoting the
    d-dimensional unit sphere and for :math:`u\in\mathbb{S}^{d-1}` denoting by
    :math:`u^*(y) = \langle u, y\rangle` the associated linear form, the
    Sliced Wasserstein distance of order :math:`p` between probability measures
    :math:`\mu,\nu` is defined as:

    .. math::
        \text{SWD}_p(\mu, \nu) = \underset{u \sim \mathcal{U}
        (\mathbb{S}^{d-1})}{\mathbb{E}}[W_p^p(u^*_\# \mu, u^*_\# \nu)]
        ^{\frac{1}{p}}

    Here, :math:`u^*_\# \mu` denotes the push-forward measure of :math:`\mu`
    by the projection :math:`u`, and :math:`W_p` the 1d Wasserstein distance
    with exponent :math:`p` for an underlying distance metric.
    In practice, the integral is approximated via a Monte-Carlo sample.

    This distance is based on [#nadjahi2020]_, the implementation based on and
    generalized from https://pythonot.github.io/gen_modules/ot.sliced.html.

    .. [#nadjahi2020]
        Nadjahi, K., De Bortoli, V., Durmus, A., Badeau, R. and Şimşekli, U.,
        2020.
        Approximate Bayesian computation with the sliced-Wasserstein distance.
        In ICASSP 2020-2020 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP) (pp. 5470-5474).
        IEEE.
    """

    def __init__(
        self,
        sumstat: Sumstat,
        metric: str = "sqeuclidean",
        p: float = 2.0,
        n_proj: int = 50,
        seed: Union[int, np.random.RandomState] = None,
        emd_1d_args: dict = None,
    ):
        """
        Parameters
        ----------
        sumstat:
            Summary statistics. Returns a ndarray of shape (n, dim), where
            n is the number of samples and dim the sample dimension.
        metric:
            Distance to use, e.g. "cityblock", "sqeuclidean", "minkowski".
        p:
            Distance exponent, to take the root in the overall distance.
            Also used in ot.emd2d_1d if "metric"=="minkowski".
        n_proj:
            Number of unit sphere projections used for Monte-Carlo
            approximation. Per projection, a one-dimensional EMD is calculated.
        seed:
            Seed used for numpy random number generation.
        emd_1d_args:
            Further keyword arguments passed on to ot.emd2_1d.
        """
        if ot is None:
            raise ImportError(
                "This distance requires the optimal transport library pot. "
                "Install via `pip install pyabc[ot]` or `pip install pot`.",
            )
        super().__init__()

        self.sumstat: Sumstat = sumstat
        self.metric: str = metric
        self.p: float = p
        self.n_proj: int = n_proj
        self.seed: Union[int, np.random.RandomState] = seed
        if emd_1d_args is None:
            emd_1d_args = {}
        self.emd_1d_args: dict = emd_1d_args

        # observed data
        self.x0: Union[dict, None] = None
        self.s0: Union[np.ndarray, None] = None

    def initialize(
        self,
        x_0: dict,
        t: int = None,
        get_sample: Callable[[], Sample] = None,
        total_sims: int = None,
    ) -> None:
        # initialize summary statistics
        self.sumstat.initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        total_sims: int,
    ) -> bool:
        # update summary statistics
        updated = self.sumstat.update(
            t=t,
            get_sample=get_sample,
            total_sims=total_sims,
        )
        if updated:
            self.s0 = self.sumstat(self.x0)
        return updated

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None,
    ) -> float:
        # compute summary statistics, shape (n, dim), (n0, dim)
        s, s0 = self.sumstat(x), self.sumstat(x_0)
        n, n0 = s.shape[0], s0.shape[0]

        dim, dim0 = s.shape[1], s0.shape[1]
        if dim != dim0:
            raise ValueError(f"Sumstat dimensions do not match: {dim}!={dim0}")

        # unit sphere samples for Monte-Carlo approximation,
        #  shape (n_proj, dim)
        sphere_samples = uniform_unit_sphere_samples(
            n_proj=self.n_proj,
            dim=dim,
            seed=self.seed,
        )

        # 1d linear projections, shape (n_proj, {n, n0})
        s_projs = np.dot(sphere_samples, s.T)
        s0_projs = np.dot(sphere_samples, s0.T)

        # weights (could also be passed/learned?)
        w, w0 = np.ones((n,)) / n, np.ones((n0,)) / n0

        # approximate integral over sphere via Monte-Carlo samples
        cost = 0.0
        for s_proj, s0_proj in zip(s_projs, s0_projs):
            # calculate optimal 1d earth mover's distance
            # this is computationally O(n*log(n)) efficient via simple sorting
            cost += ot.emd2_1d(
                x_a=s_proj,
                x_b=s0_proj,
                a=w,
                b=w0,
                metric=self.metric,
                p=self.p,
                log=False,
                **self.emd_1d_args,
            )
        cost /= self.n_proj

        # take root to match Wasserstein distance definition
        if self.p < np.inf:
            cost = cost ** (1 / self.p)

        return cost


def uniform_unit_sphere_samples(
    n_proj: int,
    dim: int,
    seed: Union[int, np.random.RandomState] = None,
) -> np.ndarray:
    r"""
    Generate uniformly distributed samples from the :math:`d-1`-dim.
    unit sphere in :math:`\mathbb{R}^d`.

    Parameters
    ----------
    n_proj: Number of samples.
    dim: Space dimension.
    seed: Seed used for numpy random number generator

    Returns
    -------
    samples:
        Vectors uniformly distributed on the unit sphere,
        shape (n_proj, dim).
    """
    if not isinstance(seed, np.random.RandomState):
        random_state = np.random.RandomState(seed)
    else:
        random_state = seed

    # generate directionally homogeneous normal samples
    projections = random_state.normal(0, 1, size=(n_proj, dim))

    # project onto sphere
    norms = la.norm(projections, ord=2, axis=1, keepdims=True)
    projections = projections / norms

    return projections
