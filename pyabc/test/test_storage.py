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
    model_names = ["fake_name_{}".format(k) for k in range(50)]
    h = History("sqlite:///" + path, )
    h.store_initial_data(0, {}, {}, {}, model_names,
                         "", "", '{"name": "pop_strategy_str_test"}')
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


def test_sum_stats_save_load(history: History):
    import scipy as sp
    arr = sp.random.rand(10)
    arr2 = sp.random.rand(10, 2)
    particle_population = [ValidParticle(0, Parameter({"a": 23, "b": 12}), .2, [.1],
                                         [{"ss1": .1, "ss2": arr2}]),
                           ValidParticle(0, Parameter({"a": 23, "b": 12}), .2, [.1],
                                         [{"ss12": .11, "ss22": arr}])
                           ]
    history.append_population(0, 42, particle_population, 2)
    weights, sum_stats = history.get_sum_stats(0, 0)
    assert (weights == 0.5).all()
    assert sum_stats[0]["ss1"] == .1
    assert (sum_stats[0]["ss2"] == arr2).all()
    assert sum_stats[1]["ss12"] == .11
    assert (sum_stats[1]["ss22"] == arr).all()


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
        model_names = ["fake_name"]*5
        h = History("sqlite:///" + path)
        h.store_initial_data(0, {}, {}, {}, model_names, "", "", "")
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


def test_population_retrieval(history):
    history.append_population(1, .23, rand_pop(0), 234)
    history.append_population(2, .123, rand_pop(0), 345)
    history.append_population(2, .1235, rand_pop(5), 20345)
    history.append_population(3, .12330, rand_pop(30), 30345)
    df = history.get_all_populations()

    assert df[df.t == 1].epsilon.iloc[0] == .23
    assert df[df.t == 2].epsilon.iloc[0] == .123
    assert df[df.t == 2].epsilon.iloc[1] == .1235
    assert df[df.t == 3].epsilon.iloc[0] == .12330

    assert df[df.t == 1].nr_samples.iloc[0] == 234
    assert df[df.t == 2].nr_samples.iloc[0] == 345
    assert df[df.t == 2].nr_samples.iloc[1] == 20345
    assert df[df.t == 3].nr_samples.iloc[0] == 30345

    assert history.alive_models(1) == [0]
    assert history.alive_models(2) == [0, 5]
    assert history.alive_models(3) == [30]


def test_population_strategy_storage(history):
    res = history.get_population_strategy()
    assert res["name"] == "pop_strategy_str_test"


def test_model_probabilities(history):
    history.append_population(1, .23, rand_pop(3), 234)
    probs = history.get_model_probabilities(1)
    assert probs.p[3] == 1
    assert probs.index.tolist() == [3]


def test_model_probabilities_all(history):
    history.append_population(1, .23, rand_pop(3), 234)
    probs = history.get_model_probabilities()
    assert (probs[3].as_matrix() == np.array([1])).all()