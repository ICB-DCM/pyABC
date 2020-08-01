import numpy as np
import pytest

from pyabc import Population
from .test_storage import rand_pop_list


def rand_pop(m: int):
    return Population(rand_pop_list(m))


def test_basic():
    m = 53
    pop = rand_pop(m)

    # call methods

    assert len(pop.get_list()) == len(pop)

    weighted_distances = pop.get_weighted_distances()
    weights, sumstats = pop.get_weighted_sum_stats()
    vals = pop.get_for_keys(
        keys=['weight', 'distance', 'parameter', 'sum_stat'])
    assert all(weighted_distances['w'] == vals['weight'])
    assert all(weighted_distances['distance'] == vals['distance'])
    assert sumstats == vals['sum_stat']
    assert all(weights == weighted_distances['w'])

    with pytest.raises(ValueError):
        pop.get_for_keys(['distance', 'w'])

    # 1 sum stat per particle in this case
    assert len(pop.get_accepted_sum_stats()) == len(pop)

    dct = pop.to_dict()
    assert m in dct.keys() and len(dct) == 1

    assert np.isclose(pop.get_model_probabilities()[m], 1)

    dst_val = -3

    def dst(*args):
        return dst_val

    pop.update_distances(dst)
    weighted_distances = pop.get_weighted_distances()
    assert all(weighted_distances['distance'] == dst_val)
