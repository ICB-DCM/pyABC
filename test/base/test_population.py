import numpy as np
import pytest

from pyabc import Population, Sample
from .test_storage import rand_pop_list


def rand_pop(m: int):
    pop = rand_pop_list(m)

    return Population(pop)


def test_basic():
    m = 53
    pop = rand_pop(m)

    # call methods

    assert len(pop.particles) == len(pop)

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

    dct = pop.get_particles_by_model()
    assert m in dct.keys() and len(dct) == 1

    assert np.isclose(pop.get_model_probabilities().loc[m, 'p'], 1)

    dst_val = -3

    def dst(*args):
        return dst_val

    pop.update_distances(dst)
    weighted_distances = pop.get_weighted_distances()
    assert all(weighted_distances['distance'] == dst_val)


def test_raises():
    """Test raises upon bad input."""
    # weights not normalized
    particles = rand_pop_list(normalized=False)
    with pytest.raises(AssertionError):
        Population(particles)

    # some not accepted
    particles = rand_pop_list()
    particles[0].accepted = False
    with pytest.raises(AssertionError):
        Population(particles)

    population = Population(rand_pop_list())
    population.particles[0].accepted = False
    with pytest.raises(AssertionError):
        Sample.from_population(population)


def test_sample():
    """Test sample construction."""
    sample = Sample(record_rejected=True, max_nr_rejected=10)
    particles = rand_pop_list(normalized=False, n_sample=100)
    for p in particles[30:]:
        p.accepted = False
    for p in particles:
        sample.append(p)

    assert len(sample.accepted_particles) == 30
    assert len(sample.rejected_particles) == 10

    # adding samples
    sample2 = Sample(record_rejected=True, max_nr_rejected=10)
    particles = rand_pop_list(normalized=False, n_sample=200)
    for p in particles[60:]:
        p.accepted = False
    for p in particles:
        sample2.append(p)

    sample = sample + sample2
    assert len(sample.accepted_particles) == sample.n_accepted == 90
    assert len(sample.rejected_particles) == 10
    assert sample.record_rejected
    assert sample.max_nr_rejected == 10

    assert not np.isclose(sum(p.weight for p in sample.accepted_particles), 1)
    sample.normalize_weights()
    assert np.isclose(sum(p.weight for p in sample.accepted_particles), 1)

    population = sample.get_accepted_population()
    assert len(population) == 90

    # zero weight
    particles = rand_pop_list(n_sample=100)
    for p in particles:
        p.weight = 0.
    sample = Sample(record_rejected=True, max_nr_rejected=10)
    for p in particles:
        sample.append(p)
    with pytest.raises(AssertionError):
        sample.normalize_weights()
