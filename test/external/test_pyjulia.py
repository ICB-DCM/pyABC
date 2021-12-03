from julia.api import Julia

# one way of making pyjulia work, see
#  https://pyjulia.readthedocs.io/en/latest/troubleshooting.html
Julia(compiled_modules=False)

# The pyjulia wrapper appears to ignore global noqas, thus per line here

import os
import tempfile

import numpy as np
import pytest

import pyabc.external.julia
from pyabc import (
    ABCSMC,
    RV,
    Distribution,
    MulticoreEvalParallelSampler,
    RedisEvalParallelSamplerServerStarter,
    Sampler,
    SingleCoreSampler,
)


def RedisEvalParallelSamplerServerStarterWrapper(**kwargs):
    kwargs.setdefault('batch_size', 5)
    kwargs.setdefault('catch', False)
    return RedisEvalParallelSamplerServerStarter(**kwargs)


@pytest.fixture(
    params=[
        SingleCoreSampler,
        MulticoreEvalParallelSampler,
        RedisEvalParallelSamplerServerStarterWrapper,
    ]
)
def sampler(request):
    s = request.param()
    try:
        yield s
    finally:
        # release all resources
        try:
            s.shutdown()
        except AttributeError:
            pass


def test_pyjulia_pipeline(sampler: Sampler):
    """Test that a pipeline with Julia calls runs through."""
    jl = pyabc.external.julia.Julia(
        source_file="doc/examples/model_julia/Normal.jl",
        module_name="Normal",
    )
    # just call it
    assert jl.display_source_ipython()  # noqa: S101

    model = jl.model()
    distance = jl.distance()
    obs = jl.observation()

    prior = Distribution(p=RV("uniform", -5, 10))

    if not isinstance(sampler, SingleCoreSampler):
        # call model once for Julia pre-combination
        distance(model(prior.rvs()), model(prior.rvs()))

    db_file = tempfile.mkstemp(suffix=".db")[1]
    abc = ABCSMC(model, prior, distance, population_size=100, sampler=sampler)
    abc.new("sqlite:///" + db_file, obs)
    abc.run(max_nr_populations=2)

    if os.path.exists(db_file):
        os.remove(db_file)


def test_pyjulia_conversion():
    """Test Julia object conversion."""
    jl = pyabc.external.julia.Julia(
        source_file="doc/examples/model_julia/Normal.jl",
        module_name="Normal",
    )
    model = jl.model()
    distance = jl.distance()
    obs = jl.observation()

    sim = model({"p": 0.5})
    assert sim.keys() == obs.keys() == {"y"}  # noqa: S101
    assert isinstance(sim["y"], np.ndarray)  # noqa: S101
    assert len(sim["y"]) == len(obs["y"]) == 4  # noqa: S101

    d = distance(sim, obs)
    assert isinstance(d, float)  # noqa: S101
