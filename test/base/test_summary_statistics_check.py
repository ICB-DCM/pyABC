"""Validate summary-statistic schemas before distances can hide mismatches."""

import pickle
import random
from decimal import Decimal
from fractions import Fraction

import cloudpickle
import numpy as np
import pytest

import pyabc
from pyabc.model import CheckedSummaryStatistics


@pytest.fixture(autouse=True)
def preserve_random_generators():
    """Keep inference examples from changing other test modules' random streams."""
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    try:
        yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def _identity(data):
    return data


def _broadcasting_distance(simulated, observed):
    return float(
        sum(
            np.sum(np.abs(simulated[key] - observed[key]) ** 2)
            for key in observed
        )
    )


def test_public_exports():
    from pyabc.model.summary_statistics import (
        CheckedSummaryStatistics as implementation,
    )

    assert pyabc.CheckedSummaryStatistics is implementation
    assert CheckedSummaryStatistics is implementation


def test_callable_is_not_applied_to_reference_and_runs_once_per_call():
    calls = []
    output = {'data': np.array([1.0, 2.0])}

    def summarize(raw):
        calls.append(raw)
        assert raw == 'raw simulation'
        return output

    checked = CheckedSummaryStatistics(summarize, {'data': np.zeros(2)})
    assert calls == []
    for _ in range(2):
        assert checked('raw simulation') is output
    assert calls == ['raw simulation', 'raw simulation']


def test_result_and_values_are_returned_without_copy_or_conversion():
    array = np.arange(6, dtype=np.int32).reshape(2, 3)
    output = {'array': array, 'number': Fraction(1, 3)}
    checked = CheckedSummaryStatistics(
        _identity, {'array': np.zeros((2, 3)), 'number': 0.0}
    )

    result = checked(output)

    assert result is output
    assert result['array'] is array
    assert result['array'].dtype == np.dtype(np.int32)
    assert result['number'] is output['number']


def test_schema_ignores_key_order_and_numeric_dtype():
    checked = CheckedSummaryStatistics(
        _identity, {'a': np.array([1, 2]), 'b': np.array([3])}
    )
    output = {'b': np.array([4.0]), 'a': np.array([5.0, 6.0])}
    assert checked(output) is output


def test_reference_schema_is_a_snapshot():
    array = np.array([1.0, 2.0])
    reference = {'array': array, 'scalar': 3.0}
    checked = CheckedSummaryStatistics(_identity, reference)

    array.resize((1, 2))
    del reference['scalar']
    reference['extra'] = np.ones(7)
    reference.clear()

    output = {'scalar': 0, 'array': np.zeros(2)}
    assert checked(output) is output
    with pytest.raises(ValueError, match='array'):
        checked({'scalar': 0, 'array': np.zeros((1, 2))})


@pytest.mark.parametrize(
    'value',
    [
        1,
        1.5,
        1 + 2j,
        np.int64(1),
        np.uint32(2),
        np.float64(3),
        np.complex128(1 + 2j),
        Fraction(1, 2),
        Decimal('0.5'),
        np.array(1.0),
    ],
)
def test_numeric_scalars_and_zero_dimensional_arrays_share_shape(value):
    checked = CheckedSummaryStatistics(_identity, {'data': value})
    output = {'data': np.array(4.0)}
    assert checked(output) is output
    assert (
        CheckedSummaryStatistics(_identity, output)({'data': value})['data']
        is value
    )


@pytest.mark.parametrize('dtype', ['int8', 'uint64', 'float32', 'complex128'])
def test_numeric_ndarrays_are_supported(dtype):
    value = np.ones((2, 3), dtype=dtype)
    output = {'data': value}
    checked = CheckedSummaryStatistics(_identity, {'data': value})
    assert checked(output) is output


def test_empty_arrays_and_empty_dictionaries_are_supported():
    empty = {}
    assert CheckedSummaryStatistics(_identity, empty)(empty) is empty
    output = {'data': np.empty((0, 2))}
    assert CheckedSummaryStatistics(_identity, output)(output) is output


def test_nan_and_infinity_do_not_add_a_value_validation_policy():
    output = {'data': np.array([np.nan, np.inf, -np.inf]), 'scalar': np.nan}
    checked = CheckedSummaryStatistics(
        _identity, {'data': np.zeros(3), 'scalar': 0.0}
    )
    assert checked(output) is output


@pytest.mark.parametrize(
    ('expected', 'actual'),
    [
        ((), (1,)),
        ((1,), ()),
        ((2,), (1, 2)),
        ((2, 3), (3, 2)),
        ((0, 2), (0, 3)),
    ],
)
def test_shape_errors_identify_key_and_both_shapes(expected, actual):
    checked = CheckedSummaryStatistics(
        _identity, {'measurement': np.zeros(expected)}
    )
    with pytest.raises(ValueError) as error:
        checked({'measurement': np.zeros(actual)})
    message = str(error.value)
    assert 'measurement' in message
    assert str(expected) in message
    assert str(actual) in message


@pytest.mark.parametrize(
    ('output', 'missing', 'extra'),
    [
        ({'a': 1}, ['b'], []),
        ({'a': 1, 'b': 2, 'extra': 3}, [], ['extra']),
        ({'replacement': 1}, ['a', 'b'], ['replacement']),
    ],
)
def test_key_errors_identify_missing_and_extra_keys(output, missing, extra):
    checked = CheckedSummaryStatistics(_identity, {'a': 0, 'b': 0})
    with pytest.raises(ValueError) as error:
        checked(output)
    for key in missing + extra:
        assert key in str(error.value)


@pytest.mark.parametrize('data', [None, [], [('a', 1)], np.array([1]), 1])
def test_reference_and_output_must_be_dictionaries(data):
    with pytest.raises(TypeError):
        CheckedSummaryStatistics(_identity, data)
    checked = CheckedSummaryStatistics(_identity, {'a': 0})
    with pytest.raises(TypeError):
        checked(data)


@pytest.mark.parametrize('key', [1, None, ('a',)])
def test_reference_and_output_keys_must_be_strings(key):
    with pytest.raises(TypeError):
        CheckedSummaryStatistics(_identity, {key: 1})
    checked = CheckedSummaryStatistics(_identity, {'a': 0})
    with pytest.raises(TypeError):
        checked({key: 1})


@pytest.mark.parametrize('summary_statistics', [None, {}, 1, 'identity'])
def test_summary_statistics_must_be_callable(summary_statistics):
    with pytest.raises(TypeError):
        CheckedSummaryStatistics(summary_statistics, {'a': 0})


@pytest.mark.parametrize(
    'value',
    [
        True,
        np.bool_(False),
        '1',
        np.str_('1'),
        np.datetime64('2026-01-01'),
        np.timedelta64(1, 'D'),
        None,
        [1, 2],
        [[1], [2, 3]],
        (1, 2),
        {'nested': 1},
        np.array([True, False]),
        np.array([1, 2], dtype=object),
        np.array([(1,)], dtype=[('field', 'i4')]),
        np.array(['text']),
        np.array([b'text']),
        np.array(['2026-01-01'], dtype='datetime64[D]'),
        np.array([1], dtype='timedelta64[D]'),
    ],
)
def test_unsupported_values_rejected_in_reference_and_output(value):
    with pytest.raises(TypeError):
        CheckedSummaryStatistics(_identity, {'data': value})
    checked = CheckedSummaryStatistics(_identity, {'data': 0})
    with pytest.raises(TypeError):
        checked({'data': value})


@pytest.mark.parametrize('kind', ['masked', 'matrix', 'memmap', 'subclass'])
def test_array_subclasses_rejected_without_coercion(kind, tmp_path):
    class ArraySubclass(np.ndarray):
        pass

    if kind == 'masked':
        value = np.ma.array([1.0, 2.0], mask=[False, True])
    elif kind == 'matrix':
        value = np.array([[1.0, 2.0]]).view(np.matrix)
    elif kind == 'memmap':
        value = np.memmap(
            tmp_path / 'array.dat', dtype=float, mode='w+', shape=(2,)
        )
    else:
        value = np.array([1.0, 2.0]).view(ArraySubclass)

    with pytest.raises(TypeError):
        CheckedSummaryStatistics(_identity, {'data': value})
    checked = CheckedSummaryStatistics(_identity, {'data': np.zeros(2)})
    with pytest.raises(TypeError):
        checked({'data': value})


def test_callable_exception_propagates_without_retry():
    calls = []
    failure = RuntimeError('simulation summary failed')

    def summarize(data):
        calls.append(data)
        raise failure

    checked = CheckedSummaryStatistics(summarize, {'data': 0})
    with pytest.raises(RuntimeError) as error:
        checked('raw')
    assert error.value is failure
    assert calls == ['raw']


@pytest.mark.parametrize('serializer', [pickle, cloudpickle])
def test_serialization_preserves_reference_snapshot(serializer):
    reference = {'a': np.array([1.0, 2.0]), 'b': 3.0}
    checked = CheckedSummaryStatistics(_identity, reference)
    reference.clear()
    restored = serializer.loads(serializer.dumps(checked))

    output = {'a': np.array([4.0, 5.0]), 'b': 6.0}
    assert restored(output) is output
    with pytest.raises(ValueError, match='a'):
        restored({'a': np.array([4.0]), 'b': 6.0})


def test_cloudpickle_supports_local_summary_callable():
    offset = 2.0
    checked = CheckedSummaryStatistics(
        lambda data: {'a': data + offset}, {'a': 0.0}
    )
    restored = cloudpickle.loads(cloudpickle.dumps(checked))
    assert restored(3.0) == {'a': 5.0}


def test_original_mismatch_is_rejected_before_broadcasting_distance():
    observed = {'a': np.array([1.0, 2.0]), 'b': np.array([3.0])}
    simulated = {'a': np.array([1.0]), 'b': np.array([2.0, 3.0])}
    # Both flatten to [1, 2, 3]; a custom distance can also broadcast silently.
    assert _broadcasting_distance(simulated, observed) == 2.0
    distance_calls = []

    def distance(x, x_0):
        distance_calls.append((x, x_0))
        return _broadcasting_distance(x, x_0)

    model = pyabc.FunctionModel(lambda pars: simulated)
    checked = CheckedSummaryStatistics(_identity, observed)
    with pytest.raises(ValueError, match='a'):
        model.accept(
            t=0,
            pars=pyabc.Parameter(),
            sum_stat_calculator=checked,
            distance_calculator=pyabc.FunctionDistance(distance),
            eps_calculator=pyabc.ConstantEpsilon(10.0),
            acceptor=pyabc.UniformAcceptor(),
            x_0=observed,
        )
    assert distance_calls == []


def test_valid_function_model_accept_preserves_statistics():
    output = {'a': np.array([1.0, 2.0]), 'b': np.array([3.0])}
    model = pyabc.FunctionModel(lambda pars: output)
    checked = CheckedSummaryStatistics(_identity, output)
    result = model.accept(
        t=0,
        pars=pyabc.Parameter(),
        sum_stat_calculator=checked,
        distance_calculator=pyabc.FunctionDistance(_broadcasting_distance),
        eps_calculator=pyabc.ConstantEpsilon(1.0),
        acceptor=pyabc.UniformAcceptor(),
        x_0=output,
    )
    assert result.sum_stat is output
    assert result.distance == 0.0
    assert result.accepted
    assert result.weight == 1.0


@pytest.mark.parametrize('valid', [True, False])
def test_single_core_abcsmc_checks_statistics_before_distance(valid, tmp_path):
    observed = {'a': np.array([0.5, 1.0]), 'b': np.array([3.0])}
    distance_calls = []

    def model(pars):
        if valid:
            return {
                'a': np.array([pars['theta'], 2 * pars['theta']]),
                'b': np.array([3.0]),
            }
        return {'a': np.array([1.0]), 'b': np.array([2.0, 3.0])}

    def distance(x, x_0):
        distance_calls.append((x, x_0))
        return _broadcasting_distance(x, x_0)

    abc = pyabc.ABCSMC(
        model,
        pyabc.Distribution(theta=pyabc.RV('uniform', 0, 1)),
        distance_function=distance,
        summary_statistics=CheckedSummaryStatistics(_identity, observed),
        eps=pyabc.ConstantEpsilon(10.0),
        population_size=3,
        sampler=pyabc.SingleCoreSampler(),
    )
    abc.new('sqlite:///' + str(tmp_path / 'checked.db'), observed)

    if valid:
        history = abc.run(max_nr_populations=1)
        assert history.n_populations == 1
        assert len(history.get_population().particles) == 3
        assert len(distance_calls) == 3
    else:
        with pytest.raises(ValueError, match='a'):
            abc.run(max_nr_populations=1)
        assert distance_calls == []
