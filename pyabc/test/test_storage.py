from pyabc import History
import pytest
import os
from pyabc.parameters import Parameter, ValidParticle
import numpy as np
import tempfile


@pytest.fixture
def history():
    # Don't use memory database for testing.
    # A real file with disconnect and reconnect is closer to the real scenario
    path = os.path.join(tempfile.gettempdir(), "history_test.db")
    h = History("sqlite:///" + path, 1, ["fake_name"])
    h.store_initial_data(0, {}, {}, {}, "", "")
    yield h
    try:
        os.remove(path)
    except FileNotFoundError:
        pass

history_2 = history


def rand_pop(m):
    pop = [ValidParticle(m, Parameter({"a": np.random.randint(10), "b": np.random.randn()}), 200, [.1], [{"ss": .1}])
        for _ in range(np.random.randint(10)+3)]
    return pop


def test_single_particle_save_load(history: History):
    particle_population = [ValidParticle(0, Parameter({"a": 23, "b": 12}), .2, [.1], [{"ss": .1}])]
    history.append_population(0, 42, particle_population, 2)

    df, w = history.weighted_parameters_dataframe(0, 0)

    assert w[0] == 1
    assert df.a.iloc[0] == 23
    assert df.b.iloc[0] == 12


def test_total_nr_samples(history: History):
    particle_population = [ValidParticle(0, Parameter({"a": 23, "b": 12}), .2, [.1], [{"ss": .1}])]
    history.append_population(0, 42, particle_population, 4234)
    history.append_population(0, 42, particle_population, 3)

    assert 4237 == history.total_nr_simulations


def test_t_count(history: History):
    particle_population = [ValidParticle(0, Parameter({"a": 23, "b": 12}), .2, [.1], [{"ss": .1}])]
    for t in range(1, 10):
        history.append_population(t, 42, particle_population, 2)
        assert t == history.max_t


def test_dataframe_storage_readout():
    path = os.path.join(tempfile.gettempdir(), "history_test.db")

    def make_hist():
        h = History("sqlite:///" + path, 5, ["fake_name"]*5)
        h.store_initial_data(0, {}, {}, {}, "", "")
        return h

    pops = {}
    histories = [make_hist() for _ in range(4)]
    for h in histories:
        for t in range(4):
            population = []
            for m in range(5):
                pops[(h, m, t)] = rand_pop(m)
                population.extend(pops[(h, m, t)])
            h.append_population(t, .1, population, 2)

    for h in histories:
        for t in range(4):
            for m in range(5):
                pop = pops[(h, m, t)]
                expected_particles_list = [p.parameter for p in pop]
                pars_df, w = h.weighted_parameters_dataframe(t, m)
                # use range(len and not zip on dataframe to not stop early
                # in case of population not completely stored
                assert np.isclose(w.sum(), 1)
                for part_nr in range(len(expected_particles_list)):
                    expected_par = expected_particles_list[part_nr]
                    actual_par = pars_df.iloc[part_nr]
                    assert expected_par.a == actual_par.a
                    assert expected_par.b == actual_par.b

    try:
        os.remove(path)
    except FileNotFoundError:
        pass
