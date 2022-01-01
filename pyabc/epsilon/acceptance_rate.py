"""Acceptance rate based optimal threshold."""

from typing import Callable, Dict, List

import autograd.numpy as anp
import numpy as np
import pandas as pd
from autograd import jacobian
from scipy import optimize

from .base import Epsilon


class AcceptanceRateEpsilon(Epsilon):
    """Optimal threshold based on predicting the acceptance rate.

    Based on [#silk2012]_.

    .. [#silk2012]
        Silk, D., Filippi, S. and Stumpf, M.P., 2012.
        Optimizing threshold-schedules for approximate Bayesian computation
        sequential Monte Carlo samplers:
        applications to molecular systems.
        arXiv preprint arXiv:1210.3296.
    """

    def __init__(self, delta: float = 0, k: float = 4.0):
        super().__init__()
        self.delta: float = delta
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
        self._update(get_all_records=get_all_records, t=t)

    def update(
        self,
        t: int,
        get_weighted_distances: Callable[[], pd.DataFrame],
        get_all_records: Callable[[], List[dict]],
        acceptance_rate: float,
        acceptor_config: dict,
    ):
        self._update(get_all_records=get_all_records, t=t)

    def configure_sampler(self, sampler):
        sampler.sample_factory.record_rejected()

    def _update(
        self,
        get_all_records: Callable[[], List[dict]],
        t: int,
    ):
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

        def acc_rate(eps: float):
            """Acceptance rate."""
            # sigmoid smooth step approximation
            acc_prob = 1.0 / (
                1.0 + anp.exp(-self.k * ((distances / eps) - 1.0))
            )
            rate = anp.sum(weights * acc_prob)
            return rate

        # objective function is 2nd derivative
        hess = jacobian(jacobian(acc_rate))

        # find maximum
        dist_max = distances.max()
        ret = optimize.minimize_scalar(
            lambda x: -hess(x),
            bounds=(0, dist_max),
            method="bounded",
        )
        eps_opt = ret.x

        # use value if acceptance rate high enough or value high enough
        acc_rate_opt = acc_rate(eps_opt)
        print(  # noqa: T001
            f"acc rate t={t}:",
            f"{eps_opt:.4e}",
            f"{acc_rate_opt:.4e}",
            f"{np.sum(weights * (distances <= eps_opt)):.4e}",
            f"{np.sum(weights * (1/(1+np.exp(-self.k*(distances - eps_opt))))):.4e}",
        )
        if acc_rate_opt > self.delta or eps_opt > distances.min():
            eps = eps_opt
        else:

            def obj(eps):
                return np.sqrt(
                    (eps / dist_max - 0) ** 2
                    + (acc_rate(eps) / acc_rate(dist_max) - 1) ** 2
                )

            ret = optimize.minimize_scalar(
                obj,
                bounds=(0, dist_max),
                method="bounded",
            )
            eps = ret.x
            print("trade-off:", eps, acc_rate(eps))  # noqa: T001

        self.eps[t] = eps

    def __call__(self, t: int) -> float:
        return self.eps[t]
