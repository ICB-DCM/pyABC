Release Notes
=============


0.3.3
-----

* Fix SGE regression. Forgot to update a module path on refactoring.


0.3.2
-----

PEP8
~~~~

Comply with PEP8 with a few exceptions where it does not make sense.
Flake8 runs now with the test. The tests do not pass if flake8 complains.


Legacy code cleanup
~~~~~~~~~~~~~~~~~~~

Remove legacy classes such as the MultivariateMultiTypeNormalDistributions
and the legacy covariance calculation. Also remove devideas folder.


0.3.1
-----

Easier usage
~~~~~~~~~~~~

Refactor the ABCSMC.set_data and provide defaults.


0.3.0
-----

Easier usage
~~~~~~~~~~~~

Provide more default values for ABCSMC. This improves usability.


0.2.0
-----

Add an efficient multicore sampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new sampler relies on forking instead of pickling for the ``sample_one``,
``simulate_one`` and ``accept_one`` functions.
This brings a huge performance improvement for single machine multicore settings
compared to ``multiprocessing.Pool.map`` like execution which repeatedly pickles.


0.1.3
-----

Initial release to the public.
