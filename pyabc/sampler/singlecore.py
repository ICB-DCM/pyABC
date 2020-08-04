from .base import Sampler
import numpy as np
from jabbar import jabbar


class SingleCoreSampler(Sampler):
    """
    Sample on a single core. No parallelization.

    Parameters
    ----------
    check_max_eval: bool
        Whether to check the maximum number of evaluations on the fly.
    """

    def __init__(
            self, check_max_eval: bool = False):
        super().__init__()
        self.check_max_eval = check_max_eval

    def sample_until_n_accepted(
            self, n, simulate_one, max_eval=np.inf, all_accepted=False):
        nr_simulations = 0
        sample = self._create_empty_sample()

        for _ in jabbar(range(n), enable=self.show_progress, keep=False):
            while True:
                if self.check_max_eval and nr_simulations >= max_eval:
                    break
                new_sim = simulate_one()
                sample.append(new_sim)
                nr_simulations += 1
                if new_sim.accepted:
                    break
        self.nr_evaluations_ = nr_simulations

        if sample.n_accepted < n:
            sample.ok = False

        return sample
