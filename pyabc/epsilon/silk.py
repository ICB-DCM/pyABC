"""Acceptance rate based optimal threshold."""

import logging
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

try:
    import autograd.numpy as anp
    from autograd import hessian
except ImportError:
    anp = hessian = None
from scipy import optimize

from .base import Epsilon

logger = logging.getLogger("ABC.Epsilon")


class SilkOptimalEpsilon(Epsilon):
    """Threshold based on the threshold - acceptance rate relation.

    Approaches based on quantiles over previously observed distances
    (:class:`pyabc.epsilon.QuantileEpsilon`) can fail to converge to the true
    posterior, by not focusing on regions in parameter space corresponding to
    small distances.
    For example when there is only a small global optimum and a large local
    optimum, are such approaches likely to focus only on the latter,
    especially for large alpha.

    In contrast, this approach is based on an estimate of the threshold -
    acceptance rate curve, aiming to balance threshold reduction with
    computational cost, and avoiding local optima.
    It is based on [#silk2012]_, but uses a simpler acceptance rate
    approximation via importance sampling propagation and automatic
    differentiation.

    It chooses a threshold maximizing the second derivative of the acceptance
    rate as a function of the threshold. This tackles in particular local
    optima, which induce a convex shape.
    In case of a convex curve, common for unimodal problems, or more general
    too low proposed thresholds or acceptance rates, instead a value is
    calculated as a trade-off between high acceptance rate and low threshold,
    minimizing a joint distance.

    .. [#silk2012]
        Silk, D., Filippi, S. and Stumpf, M.P., 2012.
        Optimizing threshold-schedules for approximate Bayesian computation
        sequential Monte Carlo samplers:
        applications to molecular systems.
        arXiv preprint arXiv:1210.3296.
        https://arxiv.org/pdf/1210.3296.pdf
    """

    def __init__(
        self,
        min_rate: float = 1e-2,
        k: float = 10.0,
    ):
        """
        Parameters
        ----------
        min_rate:
            Minimum acceptance rate. If the proposal optimal rate is lower
            (in particular for concave curves, for which the danger of local
            optima is not given), instead a trade-off of low threshold and
            high acceptance rate is performed.
        k:
            Coefficient governing the steepness of the continuous acceptance
            step aproximation (which is used to obtain a more meaningful
            2nd derivative). If inf, no continuous approximation is used.
        """
        if hessian is None:
            raise ImportError(
                "Install autograd, e.g. via `pip install pyabc[autograd]`"
            )
        super().__init__()
        self.min_rate: float = min_rate
        self.k: float = k

        self.eps: Dict[int, float] = {}

    def initialize(
        self,
        t: int,
        get_weighted_distances: Callable[[], pd.DataFrame],
        get_all_records: Callable[[], List[dict]],
        max_nr_populations: int,
        acceptor_config: dict,
    ):
        self._update(
            get_weighted_distances=get_weighted_distances,
            get_all_records=get_all_records,
            t=t,
        )

    def update(
        self,
        t: int,
        get_weighted_distances: Callable[[], pd.DataFrame],
        get_all_records: Callable[[], List[dict]],
        acceptance_rate: float,
        acceptor_config: dict,
    ):
        self._update(
            get_weighted_distances=get_weighted_distances,
            get_all_records=get_all_records,
            t=t,
        )

    def configure_sampler(self, sampler):
        # needs rejected samples to work properly
        sampler.sample_factory.record_rejected()

    def _update(
        self,
        get_weighted_distances: Callable[[], pd.DataFrame],
        get_all_records: Callable[[], List[dict]],
        t: int,
    ):
        # extract accepted particles
        dist_max = get_weighted_distances()['distance'].max()

        # extract all simulated particles
        records = get_all_records()
        records = pd.DataFrame(records)

        # previous and current transition densities
        t_pd_prev = anp.array(records['transition_pd_prev'], dtype=float)
        t_pd = anp.array(records['transition_pd'], dtype=float)
        # acceptance kernel likelihoods
        distances = anp.array(records['distance'], dtype=float)

        # compute importance weights
        weights = t_pd / t_pd_prev
        weights /= sum(weights)

        def acc_rate(eps: float, k: float = self.k):
            """Acceptance rate approximation.

            Parameters
            ----------
            eps: Acceptance threshold.
            k: Steepness of coinuous step approximation.

            Returns
            -------
            rate: Acceptance rate approximation.
            """
            # sigmoid smooth step approximation
            if k < np.inf:
                # large for distance << eps, small for distance >> eps
                acc_prob = 1.0 / (1.0 + anp.exp(k * ((distances / eps) - 1.0)))
            else:
                acc_prob = distances <= eps
            rate = anp.sum(weights * acc_prob)
            return rate

        # find optimal epsilon and corresponding acceptance rate
        eps_opt = optimal_eps_from_second_order(
            acc_rate=acc_rate,
            ub=dist_max,
        )
        acc_rate_opt = acc_rate(eps_opt)

        logger.info(
            f"Optimal threshold for t={t}: eps={eps_opt:.4e}, "
            f"estimated rate={acc_rate_opt:.4e} "
            f"(discontinuous={acc_rate(eps_opt, k=np.inf):.4e})"
        )

        # use value if acceptance rate high enough or value high enough
        if acc_rate_opt > self.min_rate or eps_opt > distances.min():
            the_eps = eps_opt
        else:
            # trade off acceptance rate and threshold value
            the_eps = tradeoff_eps(
                acc_rate=acc_rate,
                ub=dist_max,
            )
            logger.info(
                f"Overriding via trade-off: eps={the_eps}, "
                f"estimated rate={acc_rate(the_eps)} "
                f"(discontinuous={acc_rate(the_eps, k=np.inf):.4e})"
            )

        self.eps[t] = the_eps

    def __call__(self, t: int) -> float:
        return self.eps[t]


def optimal_eps_from_second_order(
    acc_rate: Callable[[float], float],
    ub: float,
):
    """Optimal epsilon maximizing the Hessian of the acceptance rates.

    Parameters
    ----------
    acc_rate: Acceptance rate function.
    ub: Upper bound on the threshold.

    Returns
    -------
    eps_opt: The optimal threshold.
    """
    # objective function is 2nd derivative
    hess = hessian(acc_rate)

    # find maximum
    ret = optimize.minimize_scalar(
        lambda x: -hess(x),
        bounds=(0, ub),
        method="bounded",
    )
    eps_opt = ret.x

    return eps_opt


def tradeoff_eps(
    acc_rate: Callable[[float], float],
    ub: float,
):
    """Find threshold trading  off low values with acceptance rate.

    Parameters
    ----------
    acc_rate: Acceptance rate function.
    ub: Upper bound on the threshold.

    Returns
    -------
    eps: The found value according to the used distance.
    """

    def obj(eps):
        """Objective function is a distance in 2-dim space."""
        return np.sqrt(
            (eps / ub - 0) ** 2 + (acc_rate(eps) / acc_rate(ub) - 1) ** 2
        )

    # minimize the distance
    ret = optimize.minimize_scalar(
        obj,
        bounds=(0, ub),
        method="bounded",
    )
    eps = ret.x

    return eps
