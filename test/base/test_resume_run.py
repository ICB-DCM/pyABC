import pytest
import numpy as np

from pyabc import (
    ABCSMC, Distribution, RV, RedisEvalParallelSamplerServerStarter,
    MulticoreEvalParallelSampler)


@pytest.fixture(params=[0, None])
def gt_model(request):
    return request.param


def RedisEvalParallelSamplerLookAheadDelayWrapper(**kwargs):
    return RedisEvalParallelSamplerServerStarter(
        look_ahead=True, look_ahead_delay_evaluation=True, **kwargs)


@pytest.fixture(params=[
    MulticoreEvalParallelSampler,
    RedisEvalParallelSamplerLookAheadDelayWrapper,
])
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


def test_resume(db_path, gt_model, sampler):
    def model(parameter):
        return {"data": parameter["mean"] + np.random.randn()}

    prior = Distribution(mean=RV("uniform", 0, 5))

    def distance(x, y):
        x_data = x["data"]
        y_data = y["data"]
        return abs(x_data - y_data)

    abc = ABCSMC(model, prior, distance, population_size=10, sampler=sampler)
    history = abc.new(db_path, {"data": 2.5}, gt_model=gt_model)
    run_id = history.id
    print("Run ID:", run_id)
    hist_new = abc.run(minimum_epsilon=0, max_nr_populations=1)
    assert hist_new.n_populations == 1

    abc_continued = ABCSMC(model, prior, distance, sampler=sampler)
    run_id_continued = abc_continued.load(db_path, run_id)
    print("Run ID continued:", run_id_continued)
    hist_contd = abc_continued.run(minimum_epsilon=0, max_nr_populations=1)

    assert hist_contd.n_populations == 2
    assert hist_new.n_populations == 2
