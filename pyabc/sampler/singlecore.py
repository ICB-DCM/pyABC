from .base import Sampler, Sample


class SingleCoreSampler(Sampler):
    """
    Sample on a single core. No parallelization.
    """

    def sample_until_n_accepted(self, sample_one, simulate_one, n):
        nr_simulations = 0
        sample = Sample()

        for _ in range(n):
            while True:
                new_param = sample_one()
                new_sim = simulate_one(new_param)
                sample.append(new_sim)
                nr_simulations += 1
                if new_sim.accepted:
                    break
        self.nr_evaluations_ = nr_simulations
        assert len(sample.accepted_particles) == n

        return sample
