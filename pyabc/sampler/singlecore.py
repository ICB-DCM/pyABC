from .base import Sampler, Sample


class SingleCoreSampler(Sampler):
    """
    Sample on a single core. No parallelization.
    """

    def sample_until_n_accepted(self, sample_options):
        nr_simulations = 0
        sample = Sample(sample_options)

        for _ in range(sample_options.n):
            while True:
                new_param = sample_options.sample_one()
                new_sim = sample_options.simulate_one(new_param)
                sample.append(new_sim)
                nr_simulations += 1
                if new_sim.accepted:
                    break
        self.nr_evaluations_ = nr_simulations
        assert len(sample.accepted_particles) == sample_options.n

        return sample
