from .base import Sampler, SamplerOptions


class SingleCoreSampler(Sampler):
    """
    Sample on a single core. No parallelization.
    """

    def sample_until_n_accepted(self, sampler_options: SamplerOptions):
        nr_simulations = 0
        sample = self._create_empty_sample()

        for _ in range(sampler_options.n):
            while True:
                new_param = sampler_options.sample_one()
                new_sim = sampler_options.simul_eval_one(new_param)
                sample.append(new_sim)
                nr_simulations += 1
                if new_sim.accepted:
                    break
        self.nr_evaluations_ = nr_simulations
        assert len(sample.accepted_particles) == sampler_options.n

        return sample
