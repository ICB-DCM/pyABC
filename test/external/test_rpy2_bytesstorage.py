import numpy as np
import pandas as pd
import pytest

from pyabc.storage.bytes_storage import from_bytes, to_bytes
from pyabc.storage.numpy_bytes_storage import _primitive_types


@pytest.fixture(
    params=[
        'empty',
        'r-df-cars',
        'r-df-faithful',
        'r-df-iris',
    ]
)
def object_(request):
    from rpy2.robjects import r

    par = request.param
    if par == 'empty':
        return pd.DataFrame()
    if par == 'df-int':
        return pd.DataFrame(
            {
                'a': np.random.randint(-20, 20, 100),
                'b': np.random.randint(-20, 20, 100),
            }
        )
    if par == 'df-float':
        return pd.DataFrame(
            {'a': np.random.randn(100), 'b': np.random.randn(100)}
        )
    if par == 'df-non_numeric_str':
        return pd.DataFrame({'a': ['foo', 'bar'], 'b': ['bar', 'foo']})

    if par == 'df-numeric_str':
        return pd.DataFrame(
            {
                'a': list(map(str, np.random.randn(100))),
                'b': list(map(str, np.random.randint(-20, 20, 100))),
            }
        )
    if par == 'df-int-float-numeric_str':
        return pd.DataFrame(
            {
                'a': np.random.randint(-20, 20, 100),
                'b': np.random.randn(100),
                'c': list(map(str, np.random.randint(-20, 20, 100))),
            }
        )
    if par == 'df-int-float-non_numeric_str-str_ind':
        return pd.DataFrame(
            {'a': [1, 2], 'b': [1.1, 2.2], 'c': ['foo', 'bar']},
            index=['first', 'second'],
        )
    if par == 'df-int-float-numeric_str-str_ind':
        return pd.DataFrame(
            {'a': [1, 2], 'b': [1.1, 2.2], 'c': ['1', '2']},
            index=['first', 'second'],
        )
    if par == 'series':
        return pd.Series({'a': 42, 'b': 3.8, 'c': 4.2})
    if par == 'series-no_ind':
        return pd.Series(np.random.randn(10))
    if par == 'py-int':
        return 42
    if par == 'py-float':
        return 42.42
    if par == 'py-str':
        return 'foo bar'
    if par == 'np-int':
        return np.random.randint(-20, 20, 100)
    if par == 'np-float':
        return np.random.randn(100)
    if par == 'np-str':
        return np.array(['foo', 'bar'])
    if par == 'np-single-int':
        return np.array(3)
    if par == 'np-single-float':
        return np.array(4.1)
    if par == 'np-single-str':
        return np.array('foo bar')
    if par == 'r-df-cars':
        return r['mtcars']
    if par == 'r-df-iris':
        return r['iris']
    if par == 'r-df-faithful':
        return r['faithful']
    raise Exception('Invalid Test DataFrame Type')


def test_storage(object_):
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import get_conversion, localconverter

    serial = to_bytes(object_)
    assert isinstance(serial, bytes)

    rebuilt = from_bytes(serial)

    # check type
    _check_type(object_, rebuilt)

    # check value
    if isinstance(object_, int):
        assert object_ == rebuilt
    elif isinstance(object_, float):
        assert object_ == rebuilt
    elif isinstance(object_, str):
        assert object_ == rebuilt
    elif isinstance(object_, np.ndarray):
        assert (object_ == rebuilt).all()
    elif isinstance(object_, pd.DataFrame):
        assert (object_ == rebuilt).all().all()
    elif isinstance(object_, pd.Series):
        assert (object_.to_frame() == rebuilt).all().all()
    elif isinstance(object_, robjects.DataFrame):
        conv = get_conversion()
        conv = conv + pandas2ri.converter
        with localconverter(conv):
            assert (conv.rpy2py(object_) == rebuilt).all().all()
    else:
        raise Exception('Could not compare')


def _check_type(object_, rebuilt):
    import rpy2.robjects as robjects

    # r objects are converted to pd.DataFrame
    if isinstance(object_, robjects.DataFrame):
        assert isinstance(rebuilt, pd.DataFrame)
    # pd.Series are converted to pd.DataFrame
    elif isinstance(object_, pd.Series):
        assert isinstance(rebuilt, pd.DataFrame)
    # <= 1 dim numpy arrays are converted to primitive type
    elif isinstance(object_, np.ndarray) and object_.size <= 1:
        for type_ in _primitive_types:
            try:
                if type_(object_) == object_:
                    assert isinstance(rebuilt, type_)
                    return
            except (TypeError, ValueError):
                pass
        raise Exception('Could not check type')
    # all others keep their type
    else:
        assert isinstance(rebuilt, type(object_))
