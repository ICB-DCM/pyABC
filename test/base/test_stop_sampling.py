import scipy.stats as st
import numpy as np
from datetime import datetime, timedelta
import pytest

from pyabc import ABCSMC, Distribution
from pyabc.sampler import MulticoreEvalParallelSampler, SingleCoreSampler


set_acc_rate = 0.2
pop_size = 10


def model(x):
    """Some model"""
    return {"par": x["par"] + np.random.randn()}


def dist(x, y):
    """Some distance"""
    return abs(x["par"] - y["par"])


def test_stop_acceptance_rate_too_low(db_path):
    """Test the acceptance rate condition."""
    abc = ABCSMC(model, Distribution(par=st.uniform(0, 10)), dist, pop_size)
    abc.new(db_path, {"par": .5})
    history = abc.run(-1, 8, min_acceptance_rate=set_acc_rate)
    df = history.get_all_populations()
    df["acceptance_rate"] = df["particles"] / df["samples"]
    assert df["acceptance_rate"].iloc[-1] < set_acc_rate
    assert df["acceptance_rate"].iloc[-2] >= set_acc_rate \
        or df["t"].iloc[-2] == -1  # calibration iteration


@pytest.fixture(
    params=[
        MulticoreEvalParallelSampler,
        SingleCoreSampler,
    ],
)
def max_eval_checked_sampler(request):
    """Samplers allowing to terminate early if generation budget exceeded."""
    s = request.param(check_max_eval=True)
    try:
        yield s
    finally:
        # release resources
        try:
            s.shutdown()
        except AttributeError:
            pass


def test_stop_early(db_path, max_eval_checked_sampler):
    """Test early stopping inside a generation."""
    sampler = max_eval_checked_sampler
    abc = ABCSMC(
        model, Distribution(par=st.uniform(0, 10)), dist,
        pop_size, sampler=sampler,
    )
    abc.new(db_path, {"par": .5})
    history = abc.run(min_acceptance_rate=set_acc_rate)
    df = history.get_all_populations()

    # offset with n_procs as more processes can have run at termination
    n_procs = sampler.n_procs if hasattr(sampler, 'n_procs') else 1
    df["corrected_acceptance_rate"] = \
        df["particles"] / (df["samples"] - (n_procs-1))

    # if already the first generation fails, the quotient is not meaningful
    assert max(df.t) == -1 or \
        df["corrected_acceptance_rate"].iloc[-1] >= set_acc_rate


def test_total_nr_simulations(db_path):
    """Test the total number of samples condition."""
    abc = ABCSMC(model, Distribution(par=st.uniform(0, 10)), dist, pop_size)
    abc.new(db_path, {"par": .5})
    max_total_nr_sim = 142
    history = abc.run(-1, 100, max_total_nr_simulations=max_total_nr_sim)
    assert history.total_nr_simulations >= max_total_nr_sim
    # Directly check on the history
    df = history.get_all_populations()
    # Make sure budget is not exceeded yet in previous iteration
    assert sum(df['samples'][:-1]) < max_total_nr_sim
    # Just to make sure .total_nr_simulations does what it's supposed to
    assert sum(df['samples']) == history.total_nr_simulations


def test_max_walltime(db_path):
    """Test the maximum walltime condition."""
    abc = ABCSMC(model, Distribution(par=st.uniform(0, 10)), dist, pop_size)
    abc.new(db_path, {"par": .5})
    init_walltime = datetime.now()
    max_walltime = timedelta(milliseconds=500)
    history = abc.run(-1, 100, max_walltime=max_walltime)
    assert datetime.now() - init_walltime > max_walltime
    assert history.n_populations < 100
