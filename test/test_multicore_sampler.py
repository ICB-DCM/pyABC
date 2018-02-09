from pyabc.sampler import (MulticoreParticleParallelSampler,
                           MulticoreEvalParallelSampler)
import pytest


class UnpickleAble:
    def __getstate__(self):
        raise Exception

    def __call__(self, *args, **kwargs):
        return True


unpickleable = UnpickleAble()


@pytest.fixture(params=[MulticoreParticleParallelSampler,
                        MulticoreEvalParallelSampler])
def sampler(request):
    return request.param()


def test_no_pickle(sampler):
    sampler.sample_until_n_accepted(unpickleable,
                                    unpickleable,
                                    10)
