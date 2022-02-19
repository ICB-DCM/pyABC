"""Tests for the `pyabc.sumstat` module."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

import pyabc
from pyabc.predictor import LinearPredictor
from pyabc.sumstat import (
    GMMSubsetter,
    IdentitySumstat,
    IdSubsetter,
    PredictorSumstat,
)
from pyabc.util import EventIxs, dict2arr, dict2arrlabels


def test_dict2arr():
    """Test conversion of dicts to arrays."""
    dct = {
        "s0": pd.DataFrame({"a": [0, 1], "b": [2, 3]}),
        "s1": np.array([4, 5]),
        "s2": 6,
    }
    keys = ["s0", "s1", "s2"]
    arr = dict2arr(dct, keys=keys)
    assert (arr == np.array([0, 2, 1, 3, 4, 5, 6])).all()

    labels = dict2arrlabels(dct, keys=keys)
    assert len(labels) == len(arr)
    assert labels == [
        "s0:a:0",
        "s0:b:0",
        "s0:a:1",
        "s0:b:1",
        "s1:0",
        "s1:1",
        "s2",
    ]

    with pytest.raises(TypeError):
        dict2arr({"s0": "alice"}, keys=["s0"])
    with pytest.raises(TypeError):
        dict2arrlabels({"s0": "alice"}, keys=["s0"])


@pytest.fixture(params=[None, [lambda x: x, lambda x: x**2]])
def trafos(request):
    """Data transformations."""
    return request.param


def test_identity_sumstat(trafos):
    """Test the IdentitySumstat."""
    sumstat = IdentitySumstat(trafos=trafos)

    x0 = {'s0': 1.0, 's1': 42.0}
    sumstat.initialize(
        t=0, get_sample=lambda: pyabc.population.Sample(), x_0=x0, total_sims=0
    )

    assert not sumstat.requires_calibration()
    assert not sumstat.is_adaptive()

    if trafos is None:
        assert (sumstat({'s1': 7.0, 's0': 3.0}) == np.array([3.0, 7.0])).all()
        assert len(sumstat.get_ids()) == 2
    else:
        assert (
            sumstat({'s1': 7.0, 's0': 3.0}) == np.array([3.0, 7.0, 9.0, 49.0])
        ).all()
        assert len(sumstat.get_ids()) == 4


def test_event_ixs():
    """Test fit index construction."""
    ixs = EventIxs(ts=1, sims=10)
    assert not ixs.act(t=0, total_sims=0)
    assert ixs.act(t=1, total_sims=0)
    assert ixs.act(t=0, total_sims=20)

    ixs = EventIxs(ts={np.inf})
    assert ixs.act(t=0, total_sims=0)
    assert ixs.act(t=7, total_sims=0)

    ixs = EventIxs(sims={10, 20})
    assert not ixs.act(t=0, total_sims=5)
    assert ixs.act(t=0, total_sims=15)
    assert not ixs.act(t=0, total_sims=16)
    assert ixs.act(t=0, total_sims=20)

    ixs = EventIxs(from_t=5)
    assert (
        not ixs.act(t=4, total_sims=50)
        and ixs.act(t=5, total_sims=50)
        and ixs.act(t=20, total_sims=50)
    )

    ixs = EventIxs(from_sims=10)
    assert (
        not ixs.act(t=4, total_sims=9)
        and ixs.act(t=4, total_sims=10)
        and ixs.act(t=4, total_sims=20)
    )


def test_pre():
    """Test chaining of summary statistics."""
    sumstat = IdentitySumstat(
        trafos=[lambda x: x**2],
        pre=IdentitySumstat(trafos=[lambda x: x, lambda x: x**2]),
    )

    assert not sumstat.requires_calibration()
    assert not sumstat.is_adaptive()
    sumstat.configure_sampler(pyabc.SingleCoreSampler())

    x0 = {'s0': 1.0, 's1': 42.0}
    sumstat.initialize(
        t=0, get_sample=lambda: pyabc.population.Sample(), x_0=x0, total_sims=0
    )

    assert (
        sumstat({'s1': 7.0, 's0': 3.0}) == np.array([3.0, 7.0, 9.0, 49.0]) ** 2
    ).all()
    assert len(sumstat.get_ids()) == 4


def test_predictor_sumstat():
    """Test predictor sumstat."""
    sumstat = PredictorSumstat(LinearPredictor(), fit_ixs={3, 5})
    assert not sumstat.requires_calibration()
    assert sumstat.is_adaptive()

    rng = np.random.Generator(np.random.PCG64(0))
    n_sample, n_y, n_p = 1000, 100, 3
    ys = rng.normal(size=(n_sample, n_y))
    ps = rng.normal(size=(n_sample, n_p))

    particles = []
    for y, p in zip(ys, ps):
        particles.append(
            pyabc.Particle(
                m=0,
                parameter=pyabc.Parameter(
                    {f"p{ix}": val for ix, val in enumerate(p)}
                ),
                sum_stat={f"s{ix}": val for ix, val in enumerate(y)},
                distance=100 + 1 * rng.normal(),
                weight=1 + 0.01 * rng.normal(),
            )
        )

    total_weight = sum(p.weight for p in particles)
    for p in particles:
        p.weight /= total_weight

    sample = pyabc.Sample.from_population(pyabc.Population(particles))
    x = particles[0].sum_stat

    # nothing should happen in initialize
    sumstat.initialize(t=0, get_sample=lambda: sample, x_0=x, total_sims=0)
    assert sumstat(x).shape == (n_y,)
    assert (sumstat(x) == ys[0]).all()
    assert len(sumstat.get_ids()) == n_y
    assert sumstat.get_ids() == [f"s{ix}" for ix in range(n_y)]

    # 3 is a fit index --> afterwards the output size should have changed
    sumstat.update(t=3, get_sample=lambda: sample, total_sims=0)
    assert sumstat(x).shape == (n_p,)
    assert len(sumstat.get_ids()) == n_p

    # change fit indices
    sumstat = PredictorSumstat(LinearPredictor(), fit_ixs={0, 1})
    sumstat.initialize(t=0, get_sample=lambda: sample, x_0=x, total_sims=0)
    assert sumstat(x).shape == (n_p,)


def test_subsetter():
    """Test whether sample cluster subsetting works."""
    rng = np.random.Generator(np.random.PCG64(0))

    n_sample, n_y, n_p = 900, 5, 2
    ss = rng.normal(size=(n_sample, n_y))
    n_sample_half = int(n_sample / 2)
    # parameter set with two modes
    par = np.row_stack(
        (
            np.array([10, 0]) + 0.1 * rng.normal(size=(n_sample_half, n_p)),
            np.array([-10, 0]) + 0.2 * rng.normal(size=(n_sample_half, n_p)),
        )
    )
    weights = rng.uniform(size=(n_sample, 1))

    # identity mapping
    subsetter = IdSubsetter()
    ss_re, par_re, weights_re = subsetter.select(x=ss, y=par, w=weights)

    assert (ss == ss_re).all()
    assert (par == par_re).all()
    assert (weights == weights_re).all()

    # Gaussian mixture model
    subsetter = GMMSubsetter(min_fraction=0.3)
    ss_re, par_re, weights_re = subsetter.select(x=ss, y=par, w=weights)
    assert n_sample * 0.6 > len(par_re) > n_sample * 0.4
    assert (par_re[0] < 0).all() or (par_re[0] > 0).all()
    assert subsetter.n_components == 2

    # 3 components
    n_sample_third = int(n_sample / 3)
    par = np.row_stack(
        (
            np.array([10, 10]) + 0.1 * rng.normal(size=(n_sample_third, n_p)),
            np.array([10, 0]) + 0.1 * rng.normal(size=(n_sample_third, n_p)),
            np.array([-10, 0]) + 0.2 * rng.normal(size=(n_sample_third, n_p)),
        )
    )
    ss_re, par_re, weights_re = subsetter.select(x=ss, y=par, w=weights)
    assert n_sample * 0.4 > len(par_re) >= n_sample * 0.3
    assert subsetter.n_components == 3

    # augment
    subsetter = GMMSubsetter(min_fraction=0.8)
    ss_re, par_re, weights_re = subsetter.select(x=ss, y=par, w=weights)
    assert len(par_re) == int(n_sample * 0.8)
    assert subsetter.n_components == 3

    # one component
    par = np.array([10, 0]) + 0.1 * rng.normal(size=(n_sample, n_p))
    ss_re, par_re, weights_re = subsetter.select(x=ss, y=par, w=weights)
    assert len(par_re) == n_sample
    assert subsetter.n_components == 1


@pytest.fixture()
def db_file():
    db_file = tempfile.mkstemp(suffix=".db")[1]
    try:
        yield db_file
    finally:
        os.remove(db_file)


def test_pipeline(db_file):
    """Test whole pipeline using a learned summary statistic."""
    rng = np.random.Generator(np.random.PCG64(0))

    def model(p):
        return {"s0": p["p0"] + 1e-2 * rng.normal(size=2), "s1": rng.normal()}

    prior = pyabc.Distribution(p0=pyabc.RV("uniform", 0, 1))

    distance = pyabc.AdaptivePNormDistance(
        sumstat=PredictorSumstat(LinearPredictor(), fit_ixs={1, 3}),
    )

    data = {"s0": np.array([0.1, 0.105]), "s1": 0.5}

    # run a little analysis
    abc = pyabc.ABCSMC(model, prior, distance, population_size=100)
    abc.new("sqlite:///" + db_file, data)
    h = abc.run(max_total_nr_simulations=1000)

    # first iteration
    df0, w0 = h.get_distribution(t=0)
    off0 = abs(pyabc.weighted_mean(df0.p0, w0) - 0.1)
    # last iteration
    df, w = h.get_distribution()
    off = abs(pyabc.weighted_mean(df.p0, w) - 0.1)

    assert off0 > off

    # alternative run with simple distance

    distance = pyabc.PNormDistance()
    abc = pyabc.ABCSMC(model, prior, distance, population_size=100)
    abc.new("sqlite:///" + db_file, data)
    h = abc.run(max_total_nr_simulations=1000)

    df_comp, w_comp = h.get_distribution()
    off_comp = abs(pyabc.weighted_mean(df_comp.p0, w_comp) - 0.1)
    assert off_comp > off

    # alternative run with info weighting
    distance = pyabc.InfoWeightedPNormDistance(
        predictor=LinearPredictor(),
        fit_info_ixs={1, 3},
    )
    abc = pyabc.ABCSMC(model, prior, distance, population_size=100)
    abc.new("sqlite:///" + db_file, data)
    h = abc.run(max_total_nr_simulations=1000)

    df_info, w_info = h.get_distribution()
    off_info = abs(pyabc.weighted_mean(df_info.p0, w_info) - 0.1)
    assert off_comp > off_info
