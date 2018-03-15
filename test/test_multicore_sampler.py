from pyabc.sampler import (MulticoreParticleParallelSampler,
                           MulticoreEvalParallelSampler,
                           SamplingOptions)
from pyabc.population import FullInfoParticle
import pytest
from multiprocessing import ProcessError


class UnpickleAble:
    def __init__(self):
        self.accepted = True

    def __getstate__(self):
        raise Exception

    def __call__(self, *args, **kwargs):
        return self


unpickleable = UnpickleAble()


def simulate_eval_one(par):
    return FullInfoParticle([], [], [], [], [], [], True)


@pytest.fixture(params=[MulticoreParticleParallelSampler,
                        MulticoreEvalParallelSampler])
def sampler(request):
    return request.param()


def test_no_pickle(sampler):
    sampling_options = SamplingOptions()
    sampling_options.n = 10
    sampling_options.sample_one = unpickleable
    sampling_options.simulate_eval_one = simulate_eval_one

    sampler.sample_until_n_accepted(sampling_options)


def raise_exception(*args):
    raise Exception("Deliberate exception to be raised in the worker "
                    "processes and to be propagated to the parent process.")


def test_exception_from_worker_propagated(sampler):
    with pytest.raises(ProcessError):
        sampling_options = SamplingOptions()
        sampling_options.n = 10
        sampling_options.sample_one = unpickleable
        sampling_options.simulate_eval_one = raise_exception

        sampler.sample_until_n_accepted(sampling_options)
