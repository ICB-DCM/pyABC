import cloudpickle as pickle
import numpy as np
import pytest

import pyabc
from pyabc.copasi import BasicoModel


@pytest.fixture(params=["stochastic", "deterministic", "directMethod", "sde"])
def method(request):
    return request.param


MAX_T = 0.1
TRUE_PAR = {"rate": 2.3}
MODEL1_PATH = "doc/examples/models/model1.xml"


def test_basic(method):
    model = BasicoModel(MODEL1_PATH, duration=MAX_T, method=method)
    data = model(TRUE_PAR)

    assert data.keys() == {"t", "X"}
    assert MAX_T / 2 < data["t"].max() <= MAX_T


def test_pickling():
    model = BasicoModel(MODEL1_PATH, duration=MAX_T, method="deterministic")

    model_re = pickle.loads(pickle.dumps(model))

    ret = model(TRUE_PAR)
    ret_re = model_re(TRUE_PAR)
    assert np.allclose(ret["t"], ret_re["t"])


def test_pipeline(db_path):
    model = BasicoModel(MODEL1_PATH, duration=MAX_T, method="deterministic")
    data = model(TRUE_PAR)
    prior = pyabc.Distribution(rate=pyabc.RV("uniform", 0, 100))

    n_test_times = 20
    t_test_times = np.linspace(0, MAX_T, n_test_times)

    def distance(x, y):
        xt_ind = np.searchsorted(x["t"], t_test_times) - 1
        yt_ind = np.searchsorted(y["t"], t_test_times) - 1
        error = (
            np.absolute(x["X"][:, 1][xt_ind] - y["X"][:, 1][yt_ind]).sum()
            / t_test_times.size
        )
        return error

    abc = pyabc.ABCSMC(model, prior, distance)
    abc.new(db_path, data)
    abc.run(max_nr_populations=3)
