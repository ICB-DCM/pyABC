from pyabc import ABCSMC, Distribution
import scipy.stats as st
import scipy as sp


def test_stop_acceptance_rate_too_low(db_path):
    set_acc_rate = 0.2

    def model(x):
        return {"par": x["par"] + sp.randn()}

    def dist(x, y):
        return abs(x["par"] - y["par"])

    abc = ABCSMC(model, Distribution(par=st.uniform(0, 10)), dist, 10)
    abc.new(db_path, {"par": .5})
    history = abc.run(-1, 8, min_acceptance_rate=set_acc_rate)
    df = history.get_all_populations()
    df["acceptance_rate"] = df["particles"] / df["samples"]
    assert df["acceptance_rate"].iloc[-1] < set_acc_rate
    assert df["acceptance_rate"].iloc[-2] >= set_acc_rate
