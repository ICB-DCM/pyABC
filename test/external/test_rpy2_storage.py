import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from pyabc import History
from pyabc.parameters import Parameter
from pyabc.population import Particle, Population


def example_df():
    return pd.DataFrame(
        {'col_a': [1, 2], 'col_b': [1.1, 2.2], 'col_c': ['foo', 'bar']},
        index=['ind_first', 'ind_second'],
    )


def path():
    return os.path.join(tempfile.gettempdir(), 'history_test.db')


@pytest.fixture(params=['file', 'memory'])
def history(request):
    # Test in-memory and filesystem based database
    if request.param == 'file':
        this_path = '/' + path()
    elif request.param == 'memory':
        this_path = ''
    else:
        raise Exception(f'Bad database type for testing: {request.param}')
    model_names = [f'fake_name_{k}' for k in range(50)]
    h = History('sqlite://' + this_path)
    h.store_initial_data(
        0, {}, {}, {}, model_names, '', '', '{"name": "pop_strategy_str_test"}'
    )
    yield h
    if request.param == 'file':
        try:
            os.remove(this_path)
        except FileNotFoundError:
            pass


def test_sum_stats_save_load(history: History):
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, r

    arr = np.random.rand(10)
    arr2 = np.random.rand(10, 2)
    particle_list = [
        Particle(
            m=0,
            parameter=Parameter({'a': 23, 'b': 12}),
            weight=0.2,
            sum_stat={
                'ss1': 0.1,
                'ss2': arr2,
                'ss3': example_df(),
                'rdf0': r['iris'],
            },
            distance=0.1,
        ),
        Particle(
            m=0,
            parameter=Parameter({'a': 23, 'b': 12}),
            weight=0.8,
            sum_stat={
                'ss12': 0.11,
                'ss22': arr,
                'ss33': example_df(),
                'rdf': r['mtcars'],
            },
            distance=0.1,
        ),
    ]

    history.append_population(
        0, 42, Population(particle_list), 2, ['m1', 'm2']
    )
    weights, sum_stats = history.get_weighted_sum_stats_for_model(0, 0)
    assert (weights == np.array([0.2, 0.8])).all()
    assert sum_stats[0]['ss1'] == 0.1
    assert (sum_stats[0]['ss2'] == arr2).all()
    assert (sum_stats[0]['ss3'] == example_df()).all().all()

    with (robjects.default_converter + pandas2ri.converter).context():
        iris_pd = robjects.conversion.get_conversion().rpy2py(r['iris'])
    assert (sum_stats[0]['rdf0'] == iris_pd).all().all()

    assert sum_stats[1]['ss12'] == 0.11
    assert (sum_stats[1]['ss22'] == arr).all()
    assert (sum_stats[1]['ss33'] == example_df()).all().all()

    with (robjects.default_converter + pandas2ri.converter).context():
        mtcars_pd = robjects.conversion.get_conversion().rpy2py(r['mtcars'])
    assert (sum_stats[1]['rdf'] == mtcars_pd).all().all()
