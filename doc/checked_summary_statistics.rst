Checking summary-statistic keys and shapes
============================================================

``CheckedSummaryStatistics`` is an optional wrapper around the callable passed
as ``ABCSMC(summary_statistics=...)``. It checks the keys and shape of each
statistic before a distance transformation can flatten them. Existing model
and distance behavior is unchanged unless this wrapper is selected.

Why check before flattening?
----------------------------

The observations ``{'a': array([1, 2]), 'b': array([3])}`` and simulations
``{'a': array([1]), 'b': array([2, 3])}`` both flatten to ``[1, 2, 3]``.
A distance on those flattened values can be zero even though the shapes of
the individual statistics differ. The wrapper raises a ``ValueError`` for
statistic ``'a'`` before distance evaluation in the ordinary model path.

Usage
-----

Use an explicit reference containing the already summarized observations::

    import numpy as np
    import pyabc

    def summarize(output):
        return output

    observed = {'a': np.array([1.0, 2.0]), 'b': np.array([3.0])}
    checked = pyabc.CheckedSummaryStatistics(summarize, observed)

    # A valid result is returned as the exact same dictionary object.
    simulated = {'b': np.array([3.0]), 'a': np.array([1.0, 2.0])}
    assert checked(simulated) is simulated

    def model(parameters):
        return {
            'a': np.array([parameters['theta'], 2.0]),
            'b': np.array([3.0]),
        }

    abc = pyabc.ABCSMC(
        model,
        pyabc.Distribution(theta=pyabc.RV('uniform', 0, 2)),
        summary_statistics=checked,
        eps=pyabc.ConstantEpsilon(0.5),
        population_size=10,
        sampler=pyabc.SingleCoreSampler(),
    )
    abc.new('sqlite://', observed)
    history = abc.run(max_nr_populations=1)

For non-identity summaries, compute the observed summary explicitly once and
pass that dictionary both to the wrapper and to ``ABCSMC.new`` or
``ABCSMC.load``. The wrapper never applies ``summarize`` to the reference.
It evaluates ``summarize`` exactly once for each simulated model output.

Schema and errors
-----------------

Both the reference and each result must be dictionaries with string keys.
Values may be numeric scalars or plain ``numpy.ndarray`` objects with integer,
unsigned-integer, floating or complex dtype. Numeric dtype changes and
dictionary order are ignored. A scalar and a zero-dimensional array both have
shape ``()``; a length-one array has the distinct shape ``(1,)``. Empty
dictionaries, empty numeric arrays and non-contiguous arrays are supported.

The wrapper stores only a snapshot of keys and shapes. Later mutation of the
reference dictionary or its arrays does not alter that snapshot. Construct a
new wrapper if the observed schema changes; it is not automatically updated
when an ``ABCSMC`` instance is created, loaded or reused.

Missing or extra keys and shape mismatches raise ``ValueError``. Shape errors
identify the key, expected shape and actual shape. Unsupported inputs raise
``TypeError``. The first implementation deliberately rejects booleans, lists,
nested dictionaries, pandas objects, object/string/structured arrays,
datetime/timedelta values and array subclasses such as masked arrays,
``matrix`` or ``memmap``. No implicit conversion is performed.

This is a structure check. It does not check finite values, physical units,
data labels, noise models, or biological meaning. The wrapped function's own
exceptions propagate unchanged. The validator does not modify its result,
but cannot prevent a user-supplied function from mutating its own inputs.

Models and serialization
------------------------

This wrapper checks the statistics returned by the ordinary ``FunctionModel``
path, including calibration through the same callable. It is separate from
the ``Sumstat`` classes used by ``Distance.sumstat``. A custom model overriding
``summary_statistics`` or ``accept`` must actually invoke the wrapper to use
its check. ``IntegratedModel`` results are not checked automatically.

Some applications intentionally use variable-length observations, for example
Wasserstein distances comparing different numbers of points. Omit the wrapper
for such representations. This optional helper is a restricted contribution
related to issue #470; it does not introduce a default check across all models.

The wrapper is a module-level class and can be serialized when its wrapped
callable can be serialized. Use an importable callable with standard
``pickle``. ``cloudpickle`` supports additional callable types where the
selected sampling backend uses it. This does not guarantee compatibility with
every distributed or external backend.
