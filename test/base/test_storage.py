import datetime
import os
import pickle
import tempfile

import numpy as np
import pandas as pd
import pyarrow
import pytest
from rpy2.robjects import pandas2ri, r
from rpy2.robjects.conversion import localconverter

import pyabc
from pyabc import History
from pyabc.parameters import Parameter
from pyabc.population import Particle, Population
from pyabc.storage.dataframe_bytes_storage import (
    DataFrameLoadException,
    df_from_bytes_csv,
    df_from_bytes_json,
    df_from_bytes_np_records,
    df_from_bytes_parquet,
    df_to_bytes_csv,
    df_to_bytes_json,
    df_to_bytes_np_records,
    df_to_bytes_parquet,
)
from pyabc.storage.df_to_file import sumstat_to_json


def example_df():
    return pd.DataFrame(
        {"col_a": [1, 2], "col_b": [1.1, 2.2], "col_c": ["foo", "bar"]},
        index=["ind_first", "ind_second"],
    )


def path():
    return os.path.join(tempfile.gettempdir(), "history_test.db")


@pytest.fixture(params=["file", "memory"])
def history(request):
    # Test in-memory and filesystem based database
    if request.param == "file":
        this_path = "/" + path()
    elif request.param == "memory":
        this_path = ""
    else:
        raise Exception(f"Bad database type for testing: {request.param}")
    model_names = ["fake_name_{}".format(k) for k in range(50)]
    h = History("sqlite://" + this_path)
    h.store_initial_data(
        0, {}, {}, {}, model_names, "", "", '{"name": "pop_strategy_str_test"}'
    )
    yield h
    if request.param == "file":
        try:
            os.remove(this_path)
        except FileNotFoundError:
            pass


@pytest.fixture
def history_uninitialized():
    # Don't use memory database for testing.
    # A real file with disconnect and reconnect is closer to the real scenario
    this_path = path()
    h = History("sqlite:///" + this_path)
    yield h
    try:
        os.remove(this_path)
    except FileNotFoundError:
        pass


def rand_pop_list(m: int = 0, normalized: bool = True, n_sample: int = None):
    """
    Create a population for model m, of random size >= 3.

    Parameters
    ----------
    m: The model index
    normalized: Whether to normalize the population weight to 1.
    n_sample: Number of samples.

    Returns
    -------
    List[Particle]: A list of particles
    """
    if n_sample is None:
        n_sample = np.random.randint(10) + 3
    pop = [
        Particle(
            m=m,
            parameter=Parameter(
                {"a": np.random.randint(10), "b": np.random.randn()}
            ),
            weight=np.random.rand() * 42,
            sum_stat={
                "ss_float": 0.1,
                "ss_int": 42,
                "ss_str": "foo bar string",
                "ss_np": np.random.rand(13, 42),
                "ss_df": example_df(),
            },
            accepted=True,
            distance=np.random.rand(),
        )
        for _ in range(n_sample)
    ]

    if normalized:
        total_weight = sum(p.weight for p in pop)
        for p in pop:
            p.weight /= total_weight

    return pop


def test_single_particle_save_load(history: History):
    particle_list = [
        Particle(
            m=0,
            parameter=Parameter({"a": 23, "b": 12}),
            weight=1.0,
            sum_stat={"ss": 0.1},
            distance=0.1,
        ),
    ]
    history.append_population(0, 42, Population(particle_list), 2, [""])

    df, w = history.get_distribution(0, 0)
    assert w[0] == 1
    assert df.a.iloc[0] == 23
    assert df.b.iloc[0] == 12


def test_save_no_sum_stats(history: History):
    """
    Test that what has been stored can be retrieved correctly
    also when no sum stats are saved.
    """
    particle_list = []
    for _ in range(0, 6):
        particle = Particle(
            m=0,
            parameter=Parameter({"th0": np.random.random()}),
            weight=1.0 / 6,
            sum_stat={"ss0": np.random.random(), "ss1": np.random.random()},
            distance=np.random.random(),
        )
        particle_list.append(particle)

    population = Population(particle_list)

    # do not save sum stats
    # use the attribute first to make sure we have no typo
    print(history.stores_sum_stats)
    history.stores_sum_stats = False

    # test some basic routines
    history.append_population(
        t=0,
        current_epsilon=42.97,
        population=population,
        nr_simulations=10,
        model_names=[""],
    )

    # just call
    history.get_distribution(0, 0)

    # test whether weights and distances returned correctly
    weighted_distances_h = history.get_weighted_distances()
    weighted_distances = population.get_weighted_distances()

    assert np.allclose(
        weighted_distances_h[['distance', 'w']],
        weighted_distances[['distance', 'w']],
    )

    weights, sum_stats = history.get_weighted_sum_stats(t=0)
    # all particles should be contained nonetheless
    assert len(weights) == len(particle_list)
    for sum_stat in sum_stats:
        # should be empty
        assert not sum_stat

    history.get_population_extended()


def test_get_population(history: History):
    population = Population(rand_pop_list(0))
    history.append_population(
        t=0,
        current_epsilon=7.0,
        population=population,
        nr_simulations=200,
        model_names=["m0"],
    )
    population_h = history.get_population(t=0)

    # length
    assert len(population) == len(population_h)

    # distances
    distances = [p.distance for p in population.particles]
    distances_h = [p.distance for p in population_h.particles]
    for d0, d1 in zip(distances, distances_h):
        assert np.isclose(d0, d1)

    # weights
    weights = [p.weight for p in population.particles]
    weights_h = [p.weight for p in population_h.particles]
    for w0, w1 in zip(weights, weights_h):
        assert np.isclose(w0, w1)


def test_single_particle_save_load_np_int64(history: History):
    # Test if np.int64 can also be used for indexing
    # This is an important test!!!
    m_list = [0, np.int64(0)]
    t_list = [0, np.int64(0)]
    particle_list = [
        Particle(
            m=0,
            parameter=Parameter({"a": 23, "b": 12}),
            weight=1.0,
            sum_stat={"ss": 0.1},
            distance=0.1,
        )
    ]
    history.append_population(0, 42, Population(particle_list), 2, [""])

    for m in m_list:
        for t in t_list:
            df, w = history.get_distribution(m, t)
            assert w[0] == 1
            assert df.a.iloc[0] == 23
            assert df.b.iloc[0] == 12


def test_sum_stats_save_load(history: History):
    arr = np.random.rand(10)
    arr2 = np.random.rand(10, 2)
    particle_list = [
        Particle(
            m=0,
            parameter=Parameter({"a": 23, "b": 12}),
            weight=0.2,
            sum_stat={
                "ss1": 0.1,
                "ss2": arr2,
                "ss3": example_df(),
                "rdf0": r["iris"],
            },
            distance=0.1,
        ),
        Particle(
            m=0,
            parameter=Parameter({"a": 23, "b": 12}),
            weight=0.8,
            sum_stat={
                "ss12": 0.11,
                "ss22": arr,
                "ss33": example_df(),
                "rdf": r["mtcars"],
            },
            distance=0.1,
        ),
    ]

    history.append_population(
        0, 42, Population(particle_list), 2, ["m1", "m2"]
    )
    weights, sum_stats = history.get_weighted_sum_stats_for_model(0, 0)
    assert (weights == np.array([0.2, 0.8])).all()
    assert sum_stats[0]["ss1"] == 0.1
    assert (sum_stats[0]["ss2"] == arr2).all()
    assert (sum_stats[0]["ss3"] == example_df()).all().all()
    with localconverter(pandas2ri.converter):
        assert (sum_stats[0]["rdf0"] == r["iris"]).all().all()
    assert sum_stats[1]["ss12"] == 0.11
    assert (sum_stats[1]["ss22"] == arr).all()
    assert (sum_stats[1]["ss33"] == example_df()).all().all()
    with localconverter(pandas2ri.converter):
        assert (sum_stats[1]["rdf"] == r["mtcars"]).all().all()


def test_total_nr_samples(history: History):
    particle_list = [
        Particle(
            m=0,
            parameter=Parameter({"a": 23, "b": 12}),
            weight=1.0,
            sum_stat={"ss": 0.1},
            distance=0.1,
        )
    ]
    population = Population(particle_list)
    history.append_population(0, 42, population, 4234, ["m1"])
    history.append_population(0, 42, population, 3, ["m1"])

    assert 4237 == history.total_nr_simulations


def test_t_count(history: History):
    particle_list = [
        Particle(
            m=0,
            parameter=Parameter({"a": 23, "b": 12}),
            weight=1.0,
            sum_stat={"ss": 0.1},
            distance=0.1,
        )
    ]
    for t in range(1, 10):
        history.append_population(t, 42, Population(particle_list), 2, ["m1"])
        assert t == history.max_t


def test_dataframe_storage_readout():
    path = os.path.join(tempfile.gettempdir(), "history_test.db")
    model_names = ["fake_name"] * 5

    def make_hist():
        h = History("sqlite:///" + path)
        h.store_initial_data(0, {}, {}, {}, model_names, "", "", "")
        return h

    pops = {}
    histories = [make_hist() for _ in range(4)]
    for h in histories:
        for t in range(4):
            particle_list = []
            for m in range(5):
                pops[(h, m, t)] = rand_pop_list(m, normalized=False)
                for particle in pops[(h, m, t)]:
                    particle_list.append(particle)
            total_weight = sum(p.weight for p in particle_list)
            for p in particle_list:
                p.weight /= total_weight
            h.append_population(
                t, 0.1, Population(particle_list), 2, model_names
            )

    for h in histories:
        for t in range(4):
            for m in range(5):
                pop = pops[(h, m, t)]
                expected_particles_list = [p.parameter for p in pop]
                pars_df, w = h.get_distribution(m, t)
                # use range(len and not zip on dataframe to not stop early
                # in case of population not completely stored
                assert np.isclose(w.sum(), 1)
                for par_ix, expected_par in enumerate(expected_particles_list):
                    actual_par = pars_df.iloc[par_ix]
                    assert expected_par.a == actual_par.a
                    assert expected_par.b == actual_par.b

    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def test_population_retrieval(history: History):
    history.append_population(
        1, 0.23, Population(rand_pop_list(0)), 234, ["m1"]
    )
    history.append_population(
        2, 0.123, Population(rand_pop_list(0)), 345, ["m1"]
    )
    history.append_population(
        2, 0.1235, Population(rand_pop_list(5)), 20345, ["m1"] * 6
    )
    history.append_population(
        3, 0.12330, Population(rand_pop_list(30)), 30345, ["m1"] * 31
    )
    df = history.get_all_populations()

    assert df[df.t == 1].epsilon.iloc[0] == 0.23
    assert df[df.t == 2].epsilon.iloc[0] == 0.123
    assert df[df.t == 2].epsilon.iloc[1] == 0.1235
    assert df[df.t == 3].epsilon.iloc[0] == 0.12330

    assert df[df.t == 1].samples.iloc[0] == 234
    assert df[df.t == 2].samples.iloc[0] == 345
    assert df[df.t == 2].samples.iloc[1] == 20345
    assert df[df.t == 3].samples.iloc[0] == 30345

    assert history.alive_models(1) == [0]
    assert history.alive_models(2) == [0, 5]
    assert history.alive_models(3) == [30]


def test_population_strategy_storage(history):
    res = history.get_population_strategy()
    assert res["name"] == "pop_strategy_str_test"


def test_model_probabilities(history):
    history.append_population(
        1, 0.23, Population(rand_pop_list(3)), 234, ["m0", "m1", "m2", "m3"]
    )
    probs = history.get_model_probabilities(1)
    assert probs.p[3] == 1
    assert probs.index.tolist() == [3]


def test_model_probabilities_all(history):
    history.append_population(
        1, 0.23, Population(rand_pop_list(3)), 234, ["m0", "m1", "m2", "m3"]
    )
    probs = history.get_model_probabilities()
    assert (probs[3].values == np.array([1])).all()


@pytest.fixture(params=[0, None], ids=["GT=0", "GT=None"])
def gt_model(request):
    return request.param


def test_observed_sum_stats(history_uninitialized: History, gt_model):
    h = history_uninitialized
    obs_sum_stats = {
        "s1": 1,
        "s2": 1.1,
        "s3": np.array(0.1),
        "s4": np.random.rand(10),
    }
    h.store_initial_data(gt_model, {}, obs_sum_stats, {}, [""], "", "", "")

    h2 = History(h.db)
    loaded_sum_stats = h2.observed_sum_stat()

    for k in ["s1", "s2", "s3"]:
        assert loaded_sum_stats[k] == obs_sum_stats[k]

    assert (loaded_sum_stats["s4"] == obs_sum_stats["s4"]).all()
    assert loaded_sum_stats["s1"] == obs_sum_stats["s1"]
    assert loaded_sum_stats["s2"] == obs_sum_stats["s2"]
    assert loaded_sum_stats["s3"] == obs_sum_stats["s3"]
    assert loaded_sum_stats["s4"] is not obs_sum_stats["s4"]


def test_model_name_load(history_uninitialized: History):
    h = history_uninitialized
    model_names = ["m1", "m2", "m3"]
    h.store_initial_data(0, {}, {}, {}, model_names, "", "", "")

    h2 = History(h.db)
    model_names_loaded = h2.model_names()
    assert model_names == model_names_loaded


def test_model_name_load_no_gt_model(history_uninitialized: History):
    h = history_uninitialized
    model_names = ["m1", "m2", "m3"]
    h.store_initial_data(None, {}, {}, {}, model_names, "", "", "")

    h2 = History(h.db)
    model_names_loaded = h2.model_names()
    assert model_names == model_names_loaded


def test_model_name_load_single_with_pop(history_uninitialized: History):
    h = history_uninitialized
    model_names = ["m1"]
    h.store_initial_data(0, {}, {}, {}, model_names, "", "", "")
    particle_list = [
        Particle(
            m=0,
            parameter=Parameter({"a": 23, "b": 12}),
            weight=1.0,
            sum_stat={"ss": 0.1},
            distance=0.1,
        )
    ]
    h.append_population(0, 42, Population(particle_list), 2, model_names)

    h2 = History(h.db)
    model_names_loaded = h2.model_names()
    assert model_names == model_names_loaded


def test_population_to_df(history: History):
    # TODO this test is not very good yet
    for t in range(3):
        for m in range(4):
            history.append_population(
                t,
                0.23,
                Population(rand_pop_list(m)),
                234,
                ["m0", "m1", "m2", "m3"],
            )
    df = history.get_population_extended(m=0)
    df_js = sumstat_to_json(df)
    assert len(df) == len(df_js)


def test_update_after_calibration(history: History):
    history.store_initial_data(None, {}, {}, {}, ["m0"], "", "", "")
    pops = history.get_all_populations()
    assert 0 == pops[pops['t'] == History.PRE_TIME]['samples'].values
    time = datetime.datetime.now()
    history.update_after_calibration(43, end_time=time)
    pops = history.get_all_populations()
    assert 43 == pops[pops['t'] == History.PRE_TIME]['samples'].values
    assert pops.population_end_time[0] == time


def test_pickle(history: History):
    pickle.dumps(history)


def test_dict_from_and_to_json():
    dct = {1: 0.5, 2: 42, 3: [1, 2, 0.1]}
    file_ = tempfile.mkstemp()[1]
    pyabc.storage.save_dict_to_json(dct, file_)
    re_dct = pyabc.storage.load_dict_from_json(file_)
    assert dct == re_dct


def test_create_db():
    # temporary file name
    file_ = tempfile.mkstemp(suffix=".db")[1]

    # set up history
    pyabc.History("sqlite:///" + file_)

    # should work just fine though file mostly empty
    pyabc.History("sqlite:///" + file_, create=False)

    # delete file and check we cannot create a History object
    os.remove(file_)
    with pytest.raises(ValueError):
        pyabc.History("sqlite:///" + file_, create=False)


def test_dataframe_formats():
    """Test correct behavior of the different dataframe storage methods."""
    df = pd.DataFrame(
        {'a': [6.57, 7], 'b': [True, False], 'c': ['hola', 'hej']},
    )

    df_parquet = df_to_bytes_parquet(df)
    df_csv = df_to_bytes_csv(df)
    df_json = df_to_bytes_json(df)

    # np does not allow object arrays
    df_float = pd.DataFrame({'a': [4.32, 5], 'b': [4, 1.24]})
    df_np_records = df_to_bytes_np_records(df_float)

    assert (df == df_from_bytes_parquet(df_parquet)).all(axis=None)
    assert (df == df_from_bytes_csv(df_csv)).all(axis=None)
    assert (df == df_from_bytes_json(df_json)).all(axis=None)
    assert (df_float == df_from_bytes_np_records(df_np_records)).all(axis=None)

    with pytest.raises(DataFrameLoadException):
        df_from_bytes_csv(df_parquet)

    # will interpret as mspack, but late pandas version dropped that method
    with pytest.raises(pyarrow.lib.ArrowInvalid):
        df_from_bytes_parquet(df_csv)


def test_export():
    """Test database export.

    Just calls export and does some very basic checks.
    """
    # simple problem
    def model(p):
        return {"y": p["p"] + 0.1 * np.random.normal()}

    prior = pyabc.Distribution(p=pyabc.RV("uniform", -1, 2))
    distance = pyabc.PNormDistance()

    try:
        # run
        db_file = tempfile.mkstemp(suffix=".db")[1]
        abc = pyabc.ABCSMC(model, prior, distance, population_size=100)
        abc.new("sqlite:///" + db_file, model({"p": 0}))
        abc.run(max_nr_populations=3)

        # export history
        for fmt in ["csv", "json", "html", "stata"]:
            out_file = tempfile.mkstemp()[1]
            try:
                pyabc.storage.export(db_file, out=out_file, out_format=fmt)
                assert os.path.exists(out_file)
                assert os.stat(out_file).st_size > 0
            finally:
                if os.path.exists(out_file):
                    os.remove(out_file)

    finally:
        if os.path.exists(db_file):
            os.remove(db_file)
