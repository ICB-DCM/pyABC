"""Optional structure checks for the ABCSMC summary-statistics callable."""

from collections.abc import Callable
from numbers import Number
from typing import Any

import numpy as np


def _check_dictionary(data: Any, source: str) -> None:
    """Require a dictionary with string keys before comparing its schema."""
    if not isinstance(data, dict):
        raise TypeError(f'{source} must be a dictionary with string keys.')
    if any(not isinstance(key, str) for key in data):
        raise TypeError(f'{source} must contain only string keys.')


def _numeric_shape(value: Any, key: str, source: str) -> tuple[int, ...]:
    """Extract a supported numeric shape without converting the value."""
    if isinstance(value, np.ndarray) and type(value) is not np.ndarray:
        raise TypeError(
            f'{source} statistic {key!r} has unsupported array subclass '
            f'{type(value).__name__}; expected a plain numeric NumPy array.'
        )
    if type(value) is np.ndarray or isinstance(value, np.generic):
        if value.dtype.kind not in 'iufc':
            raise TypeError(
                f'{source} statistic {key!r} has unsupported NumPy dtype '
                f'{value.dtype}; expected an integer, floating or complex '
                'NumPy value.'
            )
        return value.shape
    if isinstance(value, Number) and not isinstance(value, bool | np.bool_):
        return ()
    raise TypeError(
        f'{source} statistic {key!r} has unsupported type '
        f'{type(value).__name__}; expected a numeric scalar or plain '
        'numeric NumPy array. Booleans and array subclasses are not supported.'
    )


class CheckedSummaryStatistics:
    """Check dictionary keys and per-statistic shapes before distance evaluation.

    Pass this callable as ``ABCSMC(summary_statistics=...)`` to opt in. It
    validates the dictionary produced by the wrapped function before later
    distance transformations can flatten it. It is not a ``Sumstat`` subclass
    for ``Distance.sumstat`` and does not change existing defaults.

    Parameters
    ----------
    summary_statistics : callable
        Function mapping one model output to a dictionary. It is called once
        per invocation, and is never called on ``reference`` by this class.
    reference : dict
        Already summarized observations with string keys and numeric scalar
        or plain NumPy-array values. Only keys and shapes are retained, not
        the dictionary or its values. Use these same summarized observations
        as ``observed_sum_stat`` in ``ABCSMC.new`` or ``ABCSMC.load``.

    Raises
    ------
    TypeError
        If the function is not callable or the reference has unsupported keys
        or value types. Supported array dtype kinds are ``i``, ``u``, ``f``
        and ``c``. Booleans, array subclasses, lists, object/string/structured
        arrays and datetime/timedelta arrays are not supported.

    Notes
    -----
    Dictionary order and numeric dtype differences are ignored. Scalars and
    zero-dimensional arrays both have shape ``()``, distinct from ``(1,)``.
    Nonfinite values, units and data meaning are not checked. No values are
    copied, coerced or mutated by the validator, although the wrapped function
    may itself mutate its inputs.

    The schema is a snapshot: construct a new wrapper when the observations'
    schema changes. There is no automatic binding to an ``ABCSMC`` instance.
    Models overriding the usual summary-statistics/acceptance path must call
    this callable explicitly to use the check. In particular, this does not
    validate ``IntegratedModel`` results automatically.

    Serialization depends on the wrapped callable. Standard ``pickle`` works
    with suitable importable callables; ``cloudpickle`` can support additional
    callables where the selected sampling backend uses it.
    """

    def __init__(
        self, summary_statistics: Callable[[Any], dict], reference: dict
    ):
        if not callable(summary_statistics):
            raise TypeError('summary_statistics must be callable.')
        _check_dictionary(reference, 'reference')
        self._schema = tuple(
            (key, _numeric_shape(value, key, 'reference'))
            for key, value in reference.items()
        )
        self.summary_statistics = summary_statistics

    def __call__(self, model_output: Any) -> dict:
        """Evaluate the wrapped function once and validate its result.

        Parameters
        ----------
        model_output : Any
            Model output passed unchanged to the wrapped function.

        Returns
        -------
        dict
            The exact dictionary returned by the wrapped function.

        Raises
        ------
        TypeError
            If the result is not a dictionary with string keys and supported
            numeric values.
        ValueError
            If result keys or per-key shapes differ from the reference.
            Exceptions from the wrapped function propagate unchanged.
        """
        result = self.summary_statistics(model_output)
        _check_dictionary(result, 'summary_statistics result')
        expected_keys = {key for key, _ in self._schema}
        actual_keys = set(result)
        if actual_keys != expected_keys:
            raise ValueError(
                'Summary statistic keys differ from reference: '
                f'missing={sorted(expected_keys - actual_keys)!r}, '
                f'extra={sorted(actual_keys - expected_keys)!r}.'
            )
        for key, expected_shape in self._schema:
            actual_shape = _numeric_shape(
                result[key], key, 'summary_statistics result'
            )
            if actual_shape != expected_shape:
                raise ValueError(
                    f'Summary statistic {key!r} has shape {actual_shape}; '
                    f'expected {expected_shape}.'
                )
        return result
