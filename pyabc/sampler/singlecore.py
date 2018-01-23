from .base import Sampler


class SingleCoreSampler(Sampler):
    """
    Sample on a single core. No parallelization.
    """

    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        nr_simulations = 0
        simulations_all = []
        simulations_accepted = []
        for _ in range(n):
            while True:
                new_param = sample_one()
                new_sim = simulate_one(new_param)
                nr_simulations += 1
                simulations_all.append(new_sim)
                if accept_one(new_sim):
                    break
            simulations_accepted.append(new_sim)
        self.nr_evaluations_ = nr_simulations
        assert len(simulations_accepted) == n
        return {'simulations_all':simulations_all,
                'simulations_accepted':simulations_accepted}
