from pyabc import ABCSMC, Distribution
from pyabc.sampler import MulticoreEvalParallelSampler, SingleCoreSampler
import scipy.stats as st
import numpy as np


set_acc_rate = 0.2


def model(x):
    return {"par": x["par"] + np.random.randn()}


def dist(x, y):
    return abs(x["par"] - y["par"])


def test_stop_acceptance_rate_too_low(db_path):
    abc = ABCSMC(model, Distribution(par=st.uniform(0, 10)), dist, 10)
    abc.new(db_path, {"par": .5})
    history = abc.run(-1, 8, min_acceptance_rate=set_acc_rate)
    df = history.get_all_populations()
    df["acceptance_rate"] = df["particles"] / df["samples"]
    assert df["acceptance_rate"].iloc[-1] < set_acc_rate
    assert df["acceptance_rate"].iloc[-2] >= set_acc_rate


def test_stop_early(db_path):
    mc_sampler = MulticoreEvalParallelSampler(check_max_eval=True)
    sc_sampler = SingleCoreSampler(check_max_eval=True)
    for sampler in [mc_sampler, sc_sampler]:
        abc = ABCSMC(model, Distribution(par=st.uniform(0, 10)), dist, 10,
                     sampler=sampler)
        abc.new(db_path, {"par": .5})
        history = abc.run(
            max_nr_populations=8, min_acceptance_rate=set_acc_rate)
        df = history.get_all_populations()
        df["acceptance_rate"] = df["particles"] / df["samples"]
        assert df["acceptance_rate"].iloc[-1] >= set_acc_rate
