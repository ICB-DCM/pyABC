from .base import Sampler


class SingleCoreSampler(Sampler):
    """
    Sample on a single core. No parallelization.
    """

    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        nr_simulations = 0
        results = []
        for _ in range(n):
            while True:
                new_param = sample_one()
                new_sim = simulate_one(new_param)
                nr_simulations += 1
                if accept_one(new_sim):
                    break
            results.append(new_sim)
        self.nr_evaluations_ = nr_simulations
        assert len(results) == n
        return results
