import os
import pytest

import pyabc
from pyabc.sampler import (SingleCoreSampler,
                           MulticoreEvalParallelSampler,
                           RedisEvalParallelSamplerServerStarter)
import pyabc.external


def RedisEvalParallelSamplerServerStarterWrapper():
    return RedisEvalParallelSamplerServerStarter(batch_size=5)


@pytest.fixture(params=[SingleCoreSampler,
                        MulticoreEvalParallelSampler,
                        RedisEvalParallelSamplerServerStarterWrapper,
                        ])
def sampler(request):
    s = request.param()
    yield s
    try:
        s.cleanup()
    except AttributeError:
        pass


def test_r(sampler):
    r_file = "doc/examples/myRModel.R"
    r = pyabc.external.R(r_file)
    r.display_source_ipython()
    model = r.model("myModel")
    distance = r.distance("myDistance")
    sum_stat = r.summary_statistics("mySummaryStatistics")
    prior = pyabc.Distribution(meanX=pyabc.RV("uniform", 0, 10),
                               meanY=pyabc.RV("uniform", 0, 10))
    abc = pyabc.ABCSMC(model, prior, distance,
                       summary_statistics=sum_stat,
                       sampler=sampler)
    db = pyabc.create_sqlite_db_id(file_="test_external.db")
    abc.new(db, r.observation("mySumStatData"))
    history = abc.run(minimum_epsilon=0.9, max_nr_populations=2)
    history.get_weighted_sum_stats_for_model(m=0, t=1)[1][0]["cars"].head()


def test_external():
    folder = "doc/examples/rmodel/"
    executable = "Rscript"

    # initialize
    model = pyabc.external.ExternalModel(executable, folder + "model.r")
    sum_stat = pyabc.external.ExternalSumStat(
        executable, folder + "sumstat.r")
    distance = pyabc.external.ExternalDistance(
        executable, folder + "distance.r")

    # call representation function
    model.__repr__()

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
        executable="bash", file="", create_folder=False,
        suffix="sufftest", prefix="preftest")
    loc = eh.create_loc()
    assert os.path.exists(loc) and os.path.isfile(loc)
    eh.create_folder = True
    loc = eh.create_loc()
    assert os.path.exists(loc) and os.path.isdir(loc)
