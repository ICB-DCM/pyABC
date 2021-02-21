from pyabc import ABCSMC, Distribution
from pyabc.sampler import MulticoreEvalParallelSampler, SingleCoreSampler
import scipy.stats as st
import numpy as np
from datetime import datetime, timedelta


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


def test_stop_early(db_path):
    """Test early stopping inside a generation."""
    mc_sampler = MulticoreEvalParallelSampler(check_max_eval=True)
    sc_sampler = SingleCoreSampler(check_max_eval=True)
    for sampler in [mc_sampler, sc_sampler]:
        abc = ABCSMC(model, Distribution(par=st.uniform(0, 10)), dist,
                     pop_size, sampler=sampler)
        abc.new(db_path, {"par": .5})
        history = abc.run(min_acceptance_rate=set_acc_rate)
        df = history.get_all_populations()

        # offset with n_procs as more processes can have run at termination
        n_procs = sampler.n_procs if hasattr(sampler, 'n_procs') else 1
        df["corrected_acceptance_rate"] = \
            df["particles"] / (df["samples"] - (n_procs-1))

        assert df["corrected_acceptance_rate"].iloc[-1] >= set_acc_rate


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
