from julia.api import Julia

# one way of making pyjulia work, see
#  https://pyjulia.readthedocs.io/en/latest/troubleshooting.html
Julia(compiled_modules=False)

import os
import tempfile

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


def PicklingMulticoreEvalParallelSampler():
    return MulticoreEvalParallelSampler(pickle=True)


def RedisEvalParallelSamplerServerStarterWrapper(**kwargs):
    kwargs.setdefault('batch_size', 5)
    kwargs.setdefault('catch', False)
    return RedisEvalParallelSamplerServerStarter(**kwargs)


@pytest.fixture(
    params=[
        SingleCoreSampler,
        PicklingMulticoreEvalParallelSampler,
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


def test_pyjulia(sampler: Sampler):
    """Test that a pipeline with Julia calls runs through."""
    jl = pyabc.external.julia.Julia(
        source_file="doc/examples/model_julia/Normal.jl",
        module_name="Normal",
    )
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
    abc.run(max_nr_populations=5)

    if os.path.exists(db_file):
        os.remove(db_file)
