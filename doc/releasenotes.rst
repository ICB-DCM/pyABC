Release Notes
=============


0.3.0
-----

Easier usage
------------

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
