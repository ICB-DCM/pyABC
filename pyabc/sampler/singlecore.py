from .base import Sampler, Sample, SamplingOptions


class SingleCoreSampler(Sampler):
    """
    Sample on a single core. No parallelization.
    """

    def sample_until_n_accepted(self, sampling_options: SamplingOptions):
        nr_simulations = 0
        sample = Sample(sampling_options.sample_options)

        for _ in range(sampling_options.n):
            while True:
                new_param = sampling_options.sample_one()
                new_sim = sampling_options.simulate_eval_one(new_param)
                sample.append(new_sim)
                nr_simulations += 1
                if new_sim.accepted:
                    break
        self.nr_evaluations_ = nr_simulations
        assert len(sample.accepted_particles) == sampling_options.n

        return sample
