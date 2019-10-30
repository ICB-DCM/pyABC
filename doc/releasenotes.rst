.. _releasenotes:

Release Notes
=============


0.9 series
..........


0.9.20 (2019-10-30)
-------------------

* Add high level versions of the kde plotting routines (#204).


0.9.19 (2019-10-23)
-------------------

* Move to cffi>=1.13.1 after that bug was surprisingly quickly fixed (#195).
* Create sub-module for epsilon (#189).
* Add plots for sample and acceptance rate trajectories (#193).


0.9.18 (2019-10-20)
-------------------

* Add create_sqlite_db_id convenience function to create database names.
* Temporarily require cffi=1.12.2 for rpy2 on travis (all #185).
* Introduce UniformAcceptor and SimpleFunctionAcceptor classes to streamline
  the traditional acceptance step.
* Add AcceptorResult and allow weights in the acceptance step (all #184).


0.9.17 (2019-10-10)
-------------------

* Use latest pypi rpy2 version on travis and rtd since now the relevant
  issues were addressed there (easier build, esp. for users).
* Update rtd build to version 2 (all #179).
* Render logo text for platform independence.
* Prevent stochastic transition test from failing that often.
* Remove deprecated pd.convert_objects call in web server.
* Allow pandas.Series as summary statistics, by conversion to
  pandas.DataFrame (all #180).


0.9.16 (2019-10-08)
-------------------

* Add AggregatedDistance function, and a basic self-tuned version
  AdaptiveAggregatedDistance.
* Add additional factors to PNormDistance and AggregatedDistance for
  flexibility. Minor API break: argument w renamed to weights.
* In the adaptive_distances and the aggregated_distances notebooks, add
  examples where some methods can fail.
* Add plot_total_sample_numbers plot (all #173).


0.9.15 (2019-09-15)
-------------------

* Some extensions of external simulators interface (#168).
* Add basic plots of summary statistics (#165).
* Document high-performance infrastructure usage (#159).
* Self-administrative: Add social preview (#158), and link to zenodo (#157).
* Fix external deprecations (#153).
* Re-add R related tests (#148).


0.9.14 (2019-08-08)
-------------------

* Update to rpy2 3.1.0 (major change) (#140).
* pandas data frames saved in database via pyarrow parquet, no longer
  msgpack (deprecated), with backward compatibility for old databases (#141).
* Redis workers no longer stop working when encountering model errors (#133).
* Minor edits, esp. color, size, axes options to plotting routines.


0.9.13 (2019-06-25)
-------------------

* Fix dependency updates (rpy2, sklearn) and travis build.
* Add option to limit number of particles for adaptive distance updates.
* Rename confidence -> credible intervals and plots (Bayesian context).
* Extract from database and plot reference parameter values.
* Allow to plot MAP value approximations in credible interval plots.
* Add a general interface to external scripts that allow using pyabc in a
  simple way in particular with other programing languages.
 

0.9.12 (2019-05-02)
-------------------

* Reorganize distance module (minor API change:
  distance_functions -> distance, and some classes shortened accordingly)
* Allow to pass parameters to Acceptor and Distance.
* Make time and parameter arguments to distance functions optional.
* Rewrite lazy evaluation for calibration sample in ABCSMC class.
* Give default values for ABCSMC.run arguments, which set no stopping
  criterion.
* Add function and plot for effective sample size.


0.9.11 (2019-04-01)
-------------------

* Run some notebooks as part of the tests.
* Automatize pypi upload via travis.


0.9.10 (2019-03-27)
-------------------

* Save number of samples taken in calibration step in database.
* Fix error with reported number of simulations in EpsMixin based samplers.
* Fix several warnings.


0.9.9 (2019-03-25)
------------------

* Monitor code quality using codacy and codecov.
* Extend visualization routines: Add histogram, sample number, epsilon
  trajectory, model probability, and credible interval plots.
* Test visualization routines on travis.
* Fix problem with the History.get_weighted_distances function after update to
  sqlalchemy>=1.3.0.
* Add random walk based transition for discrete parameters.


0.9.8 (2019-02-21)
------------------

* Tidy up returning of rejected samples in Sample (not only summary
  statistics).
* Recreate a population from file in History.get_population().
* Speed up loading from database by eager loading.
* Document the change of the contribution scheme to master+develop.


0.9.7 (2019-02-20)
------------------

* Allow for the database to save no summary statistics for testing purposes.
* Tidy up some pyabc.History methods.
* pyabc.History.id set by default to the largest index (previously 0),
  corresponding to the latest inserted analysis.


0.9.6 (2019-02-01)
------------------

* Fix several errors with the readthedocs (rtd) documentation.
* Speed-up rtd build by removing unnecessary conda and pip requirements.
* Clean-up requirements for travis and rtd.
* Change rtd design from alabaster to sphinx_rtd_theme since it implements
  better navigation.


0.9.5 (2019-01-17)
------------------

* ABCSMC can pass observed summary statistics to distance functions
  (required for some scale functions, and to make the
  methods robust to volatile summary statistics).
* Implementation of more scale functions (distance_functions.scales), in
  particular some taking into account the bias to the observed data.
* AdaptivePNormDistance accepts a Callable as scaling scheme, allowing
  for more flexibility.


0.9.4 (2018-12-18)
------------------

* Can specify kde and number of bins for all visualization routines.
* Can re-submit observed sum stats to ABCSMC.load() function in case
  it cannot be read correctly from the db.


0.9.3 (2018-12-01)
------------------

* Fix serious memory problem resulting from pickling more than necessary
  for parallel sampling.
* Update logo, readme.
* Make tidying optional in abc-export (default behavior not changed).


0.9.2 (2018-09-10)
------------------

* Minor error and warning fixes due to API changes in pandas, seaborn (not
  used any more), and change of the R installation on travis.


0.9.1 (2018-06-05)
------------------

* Default visualizations like plot_kde_matrix() can plot reference values,
  useful for testing purposes.


0.9.0
-----

* Acceptance transferred to an Acceptor object to allow for more
  flexibility (i.e. not only on a single comparison as per default).
* This acceptor is passed to the ABCSMC object.
* Update of distance and epsilon synchronized after each iteration and moved
  to update() methods.
* initialize() for DistanceFunction and Epsilon also called in load() method,
  given a time point to initialize for, and made optional via a
  require_initialize flag. This makes sure these objects are always correctly
  initialized.
* PNormDistance and AdaptivePNormDistance (prev. WeightedPNormDistance)
  improved to allow for more customization.
* ABCSMC.set_data() method removed.
* API breaks for DistanceFunction, Epsilon, Model.


0.8 series
..........


0.8.21
------

* Implementation of adaptive distances feature. Distance functions can adapt
  via an update() method.
* In particular add WeightedPNormDistance (special case:
  WeightedEuclideanDistance). Also add non-weighted versions.
* Simplify Sampler.sample_until_n_accepted interface.
* Extend Sampler class to allow for customization, e.g. by the distance
  functions.
* Generalize MedianEpsilon to QuantileEpsilon.
* Make Viserver work with latest bokeh version.


0.8.20
------

* Add batch sampling now also to the REDIS evaluation parallel sampler
  (dynamic scheduling)


0.8.19
------

* Bug fix. Fix a race condition in the redis evaluation parallel sampler
  (dynamic scheduling). An error occured if a worker tried to start to work
  on a population after the other workers had already terminated the
  population.


0.8.18
------

* Minor bug fix. Ensure that the multicore samplers raise an Exception if
  an Exception occurs in the worker processes.
* Clarify that weighted distances are not normalized in case of having more
  than a single simulation per proposed parameter.
  Also add corresponding tests.
* Add n_worker method to the RedisEvalParallelSampler to enable querying of
  the number of connected workers.
* Add in-memory database support. Useful, e.g., for benchmarking on slow
  filesystems or with rather slow network connections.


0.8.17
------

Make git and gitpython an optional dependency.


0.8.16
------

* Add "abc-redis-manager reset-workers" command in case workers were
  unexpectedly killed.
* Adapt web server to changed bkcharts API.


0.8.15
------

* Bug fix. Rand seed initialization in case of starting multiple workers
  with --processes in redis server was not correct.


0.8.14
------

* Bug fix in MulticoreEvalParallelSampler. The multiprocessing.Queue could fill
  up and cause a deadlock on joining the workers. This is now fixed.
* Rename ``population_specification`` to ``population_size``.
* Improve ``plot_kde_matrix`` plot ranger are now handled in a less confusing
  way

0.8.13
------

* Minor doc fixes
* Python 3.5 support dropped. It might still work for a while with Python 3.5
  but this is not guaranteed anymore.
* Add kde matrix visualization function
* Add 2d tumor growth example
* Add Gillespie example
* Change license


0.8.12
------

* Minor bug fix. Visualization server produced error when JSON information
  was empty.
* Adapt to new bkcharts packge.


0.8.11
------

Ensure R source file is reloaded when unpickling R objects.


0.8.10
------

Add ``--id`` option to abc-export to handle databases with multiple ABC runs.


0.8.9
-----

Ensure that summary statistics have names.
Also add kwargs to ``plot_kde_2d`` which are passed to pcolormesh.

0.8.8
-----

Add ``--processes`` option to abc-redis-worker to start a number of workers
in parallel.


0.8.7
-----

Make rpy2 an optional dependency. If rpy2 is installed, then R can be used
if not, the rest will still work.

0.8.6
-----

minor bug fixes

0.8.5
-----

* minor bug fix in plot_kde_2d if the axis is provided


0.8.5
-----

* minor bug fix. The external.R interface did not display the source code
  correctly.
* minor doc updates


0.8.4
-----

* support serialization of DataFrames used as summary statistics for storage
  in the database. This feature is still considered experimental.
* Add command line utility to export pyABC's database to different file formats
  such as csv, feather, html, json and more.


0.8.3
-----

* Add (experimental) support for models defined in R.
* Add some visualization functions for convenience.


0.8.2
-----

Bug fixes for web server.


0.8.1
-----

Minor internal refactorings and minor documetation updates.
Nothing a user should notice.

0.8.0
-----

* Deprecate the "set_data" method of the ABCSMC class.
  Use the "new" method instead.
* Add a "load" method to the ABCSMC class for easier resuming stored ABCSMC
  runs.
* Add an example to the documentation how to resume stored ABC-SMC runs.
* Rename the acceptance_rate parameter form ABCSMC.run to min_acceptance_rate
  for clarity. Usage of acceptance_rate is deprecated.
* Various documentation improvements, correcting typos, clarifications, etc.


0.7 series
..........


0.7.2
-----

Easier early stopping models via the IntegratedModel class.
Also has now examples.


0.7.1
-----


* Minor refactoring for better Windows compatibility. But runs in serial
  on Windows


0.7.0
-----

* ABCSMC.run gets a new parameter "acceptance_rate" to stop sampling if the
  acceptance rate drops too low.
* History.get_all_populations returns a DataFrame with columns "t",
  "population_end_time", "samples", "epsilon", "particles". That is
  "nr_samples" got renamed to "samples" and "particles" is new.


0.6 series
..........


0.6.4
-----

Performance improvement. Use MulticoreEvalParallelSampler as default. This
should bring better performance for machines with many cores and comparatively
small population sizes.

0.6.3
-----

Bug fix. Ensure numpy.int64 can also be passed to History methods were an
integer argument is expected.


0.6.2
-----

Bug fix. Forgot to add the new Multicore base class.


0.6.1
-----

MulticoreEvalParallelSampler gets an n_procs parameter.


0.6.0
-----

History API
~~~~~~~~~~~

Change the signature from History.get_distribution(t, m)
to History.get_distribution(m, t) and make the time argument optional
defaulting to the last time point


0.5 series
..........


0.5.2
-----

* Minor History API changes
    * Remove History.get_results_distribution
    * rename History.get_weighted_particles_dataframe to
      History.get_distribution


0.5.1
-----

* Minor ABCSMC API changes
    * Mark the de facto private methods as private by prepending an
      underscore. This should not cause trouble as usually noone would
      ever use these methods.


0.5.0
-----

* Usability improvements and minor API canges
    * ABCSMC accepts now an integer to be passed for constant population size
    * The maximum number populations specification has moved from the
      PopulationStrategy classes to the ABCSMC.run method. The ABCSMC.run
      method will be where it is defined when to stop.


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
