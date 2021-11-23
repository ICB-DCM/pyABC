import os

import pytest

import pyabc
import pyabc.external
from pyabc.sampler import (
    MulticoreEvalParallelSampler,
    RedisEvalParallelSamplerServerStarter,
    SingleCoreSampler,
)


def RedisEvalParallelSamplerServerStarterWrapper():
    return RedisEvalParallelSamplerServerStarter(batch_size=5)


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


def test_external():
    folder = "doc/examples/model_r/"
    executable = "Rscript"

    # initialize
    model = pyabc.external.ExternalModel(executable, folder + "model.r")
    sum_stat = pyabc.external.ExternalSumStat(executable, folder + "sumstat.r")
    distance = pyabc.external.ExternalDistance(
        executable, folder + "distance.r"
    )

    # call representation function
    print(model.__repr__())

    # create a dummy observed sum stat
    dummy_sum_stat = pyabc.external.create_sum_stat()

    pars = {'meanX': 3, 'meanY': 3.5}

    # call model
    m = model(pars)
    # call sumstat
    s = sum_stat(m)
    # call distance
    distance(s, dummy_sum_stat)


def test_external_handler():
    eh = pyabc.external.ExternalHandler(
        executable="bash",
        file="",
        create_folder=False,
        suffix="sufftest",
        prefix="preftest",
    )
    loc = eh.create_loc()
    assert os.path.exists(loc) and os.path.isfile(loc)
    eh.create_folder = True
    loc = eh.create_loc()
    assert os.path.exists(loc) and os.path.isdir(loc)
