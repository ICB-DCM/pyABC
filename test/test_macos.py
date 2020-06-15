"""Test some basic functionality of pyabc on mac."""

import pytest
import numpy as np

import pyabc


@pytest.fixture(params=[lambda: None,
                        pyabc.SingleCoreSampler,
                        pyabc.MulticoreEvalParallelSampler,
                        pyabc.MulticoreParticleParallelSampler,
                        ])
def sampler(request):
    s = request.param()
    yield s
    try:
        s.cleanup()
    except AttributeError:
        pass


def test_basic(sampler: pyabc.sampler.Sampler):
    """Some basic tests."""
    def model(par):
        return {'s0': par['p0'] + np.random.randn(4)}

    def distance(x, y):
        return np.sum(x['s0'] - y['s0'])

    x0 = model({'p0': 2})
    prior = pyabc.Distribution(p0=pyabc.RV("uniform", 0, 10))

    abc = pyabc.ABCSMC(
        model, prior, distance, sampler=sampler, population_size=50)
    abc.new(pyabc.create_sqlite_db_id(), x0)
    abc.run(max_nr_populations=4)
