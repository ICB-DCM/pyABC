from pyabc import History
import pytest
import os
from pyabc.storage import ValidParticle
from pyabc.parameters import Parameter


@pytest.fixture
def history():
    path = "/tmp/history_test.db"
    h = History("sqlite:///" + path, 1, ["fake_name"])
    h.store_initial_data(0, {}, {}, {}, "", "")
    yield h
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def test_single_particle_save_load(history: History):
    particle_population = [ValidParticle(0, Parameter({"a": 23, "b": 12}), .2, [.1], [{"ss": .1}])]
    history.append_population(0, 42, particle_population, 2)

    a, wa = history.get_distribution(0, 0, "a")
    b, wb = history.get_distribution(0, 0, "b")

    assert a[0] == 23
    assert wa[0] == 1

    assert b[0] == 12
    assert wb[0] == 1