from .base import Sampler


class SingleCoreSampler(Sampler):
    """
    Sample on a single core. No parallelization.
    """

    def sample_until_n_accepted(self, n, simulate_one):
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
        if sample.n_accepted != n:
            raise AssertionError(
                "The number of accepted samples is not as expected.")

        return sample
