from .base import Sampler


class SingleCoreSampler(Sampler):
    """
    Sample on a single core. No parallelization.
    """

    def sample_until_n_accepted(self, n, simulate_one, all_accepted=False):
        nr_simulations = 0
        sample = self._create_empty_sample()

        for _ in range(n):
            while True:
                new_sim = simulate_one()
                sample.append(new_sim)
                nr_simulations += 1
                if new_sim.accepted:
                    break
        self.nr_evaluations_ = nr_simulations

        return sample
