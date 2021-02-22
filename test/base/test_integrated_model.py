import numpy as np
import pyabc


def simulate(n_steps, step_size):
    trajectory = np.zeros(n_steps)
    for t in range(1, n_steps):
        xi = np.random.uniform()
        trajectory[t] = trajectory[t-1] + xi * step_size
    return trajectory


class MyStochasticProcess(pyabc.IntegratedModel):

    def __init__(self, n_steps, gt_step_size, gt_trajectory, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_steps = n_steps
        self.gt_step_size = gt_step_size
        self.gt_trajectory = gt_trajectory
        self.n_early_stopped = 0

    def integrated_simulate(self, pars, eps):
        cumsum = 0
        trajectory = np.zeros(self.n_steps)
        for t in range(1, self.n_steps):
            xi = np.random.uniform()
            next_val = trajectory[t-1] + xi * pars["step_size"]
            cumsum += abs(next_val - self.gt_trajectory[t])
            trajectory[t] = next_val
            if cumsum > eps:
                self.n_early_stopped += 1
                return pyabc.ModelResult(accepted=False)

        return pyabc.ModelResult(accepted=True,
                                 distance=cumsum,
                                 sum_stat={"trajectory": trajectory})


def test_early_stopping():
    """Basic test whether an early stopping pipeline works.
    Heavily inspired by the `early_stopping` notebook.
    """
    prior = pyabc.Distribution(step_size=pyabc.RV("uniform", 0, 10))

    n_steps = 30
    gt_step_size = 5
    gt_trajectory = simulate(n_steps, gt_step_size)

    model = MyStochasticProcess(n_steps=n_steps, gt_step_size=gt_step_size,
                                gt_trajectory=gt_trajectory)

    abc = pyabc.ABCSMC(
        models=model,
        parameter_priors=prior,
        distance_function=pyabc.NoDistance(),
        population_size=30,
        transitions=pyabc.LocalTransition(k_fraction=.2),
        eps=pyabc.MedianEpsilon(300, median_multiplier=0.7),
    )
    # initializing eps manually is necessary as we only have an integrated
    #  model
    # TODO automatically iniitalizing would be possible, e.g. using eps = inf

    abc.new(pyabc.create_sqlite_db_id())
    abc.run(max_nr_populations=3)
