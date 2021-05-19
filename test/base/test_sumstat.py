"""Tests for the `pyabc.sumstat` module."""

import pytest
import numpy as np
import os
import tempfile

import pyabc
from pyabc.sumstat import IdentitySumstat, PredictorSumstat
from pyabc.predictor import LinearPredictor


@pytest.fixture(params=[None, [lambda x: x, lambda x: x**2]])
def trafos(request):
    """Data transformations."""
    return request.param


def test_identity_sumstat(trafos):
    """Test the IdentitySumstat."""
    sumstat = IdentitySumstat(trafos=trafos)

    x0 = {'s0': 1., 's1': 42.}
    sumstat.initialize(
        t=0, get_sample=lambda: pyabc.population.Sample(), x_0=x0)

    if trafos is None:
        assert (sumstat({'s1': 7., 's0': 3.}) == np.array([3., 7.])).all()
    else:
        assert (sumstat({'s1': 7., 's0': 3.}) == np.array([3., 7., 9., 49.]))\
            .all()


def test_pre():
    """Test chaining of summary statistics."""
    sumstat = IdentitySumstat(
        trafos=[lambda x: x**2],
        pre=IdentitySumstat(trafos=[lambda x: x, lambda x: x**2]))

    x0 = {'s0': 1., 's1': 42.}
    sumstat.initialize(
        t=0, get_sample=lambda: pyabc.population.Sample(), x_0=x0)

    assert (sumstat({'s1': 7., 's0': 3.}) == np.array([3., 7., 9., 49.])**2)\
        .all()


def test_predictor_sumstats():
    """Test predictor sumstat."""
    sumstat = PredictorSumstat(LinearPredictor(), fit_ixs={3, 5})

    rng = np.random.Generator(np.random.PCG64(0))
    n_sample, n_y, n_p = 1000, 100, 3
    ys = rng.normal(size=(n_sample, n_y))
    ps = rng.normal(size=(n_sample, n_p))

    particles = []
    for y, p in zip(ys, ps):
        particles.append(pyabc.Particle(
            m=0,
            parameter=pyabc.Parameter({f"p{ix}": val
                                       for ix, val in enumerate(p)}),
            sum_stat={f"s{ix}": val for ix, val in enumerate(y)},
            distance=100 + 1 * rng.normal(),
            weight=1 + 0.01 * rng.normal(),
        ))

    total_weight = sum(p.weight for p in particles)
    for p in particles:
        p.weight /= total_weight

    sample = pyabc.Sample.from_population(pyabc.Population(particles))
    x = particles[0].sum_stat

    # nothing should happen in initialize
    sumstat.initialize(t=0, get_sample=lambda: sample, x_0=x)
    assert sumstat(x).shape == (n_y,)
    assert (sumstat(x) == ys[0]).all()

    # 3 is a fit index --> afterwards the output size should have changed
    sumstat.update(t=3, get_sample=lambda: sample)
    assert sumstat(x).shape == (n_p,)

    # change fit indices
    sumstat = PredictorSumstat(LinearPredictor(), fit_ixs={0, 1})
    sumstat.initialize(t=0, get_sample=lambda: sample, x_0=x)
    assert sumstat(x).shape == (n_p,)


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
        sumstat=PredictorSumstat(LinearPredictor()),
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

    # last iteration
    df_comp, w_comp = h.get_distribution()
    off_comp = abs(pyabc.weighted_mean(df_comp.p0, w_comp) - 0.1)

    assert off_comp > off
