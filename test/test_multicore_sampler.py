from parallel import MulticoreSampler


class UnpickleAble:
    def __getstate__(self):
        raise Exception

    def __call__(self, *args, **kwargs):
        return True


unpickleable = UnpickleAble()


def test_no_pickle():
    sampler = MulticoreSampler()
    sampler.sample_until_n_accepted(unpickleable, unpickleable, unpickleable, 10)
