Release Notes
=============

0.4 series
..........


0.4.4
-----

* Improvements to adaptive population size strategy
   * Use same CV estimation algorithm for Transition and PopulationStrategy
   * Bootstrapping on full joint space for model selection


0.4.3
-----

* Fix edge case of models without parameters for population size adaptation


0.4.2
-----

* Changes to the experimental adaptive population strategy.
   * Smarter update for model selection
   * Better CV estimation



0.4.1
-----

* fix minor bug in RVs wrapper. args and keyword args were not passed to the
  wrapper random variable.


0.4.0
-----

* Add local transition class which makes a local KDE fit.
* Fix corner cases of adaptive population size strategy
* Change the default: Do not stop if only a single model is alive.
* Also include population 0, i.e. a sample from the prior, in the websever
  visualization
* Minor bug fixes
    * Fix inconsistency in ABC options if db_path given as sole string argument
* Add four evaluation parallel samplers
    * Dask based implementation
        * More communication overhead
    * Future executor evaluation parallel sampler
        * Very similar to the Dask implementation
    * Redis based implementation
        * Less communication overhad
        * Performs also well for short running simulations
    * Multicore evaluation parallel sampler
        * In most common cases, where the population size is much bigger
          than the number of cores, this sampler is not going to be faster
          than the multicore particle parallel sampler.
        * However, on machines with lots of cores and moderate sized populations
          this sampler might be faster


0.3 series
..........

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


0.2 series
..........

0.2.0
-----

Add an efficient multicore sampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new sampler relies on forking instead of pickling for the ``sample_one``,
``simulate_one`` and ``accept_one`` functions.
This brings a huge performance improvement for single machine multicore settings
compared to ``multiprocessing.Pool.map`` like execution which repeatedly pickles.


0.1 series
..........

0.1.3
-----

Initial release to the public.
