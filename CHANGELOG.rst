.. _releasenotes:

Release Notes
=============


0.12 Series
...........

0.12.16 (2025-05-23)
--------------------

General:

* Migration to PEtab v2.


0.12.15 (2024-10-29)
--------------------

General:

* Minor improvements in the documentation
* Minor bug fixes in dependencies
* Minor bug fixed in general
* Dropping support of python 3.9
* Adding support for python 3.12


0.12.14 (2023-11-10)
--------------------

Visualization:

* Selected plotly versions of matplotlib visualizations

General:

* Added functionality to evaluate the model using boundary values of parameter


0.12.13 (2023-11-08)
--------------------

Fixes (#615):

* Fix for wrong assets path in abc-server-dash

General

* Added new maintainers to RTD's about section


0.12.12 (2023-08-18)
--------------------

Visualization:

* Dash-based visualization server (#581)


0.12.11 (2023-07-06)
--------------------

Fixes (#608):

* Fix petab test suite (different name resolution)
* Fix LocalTransition (pandas -> numpy argument)
* Fix sklearn intersphinx


0.12.10 (2023-05-09)
--------------------

General:

* Update to Python 3.11, drop Python 3.8 support
* Turn simple files into submodules for cleaner import
* Fix dask sampler (change default pickling and use global function that it can pickle)

Documentation:

* Configure readthedocs build environment
* Refactor API docs (add [+]; easier navigation)
* Correct scipy intersphinx link
* Add docstrings to submodules

Minor fixes:

* Flake8 fixes of dict creation from key+value
* Fix imports from correct submodule in external submodule
* Fix package versions in migration test

Visualization:

* Contour plots as a direct alternative to kernel density heat maps
  and histograms (#597)
* Fix column renaming in visserver


0.12.9 (2023-03-01)
-------------------

Minor:

* Improve documentation of p-norm (#592)
* Update citation to JOSS
* Temporarily fixate sqlalchemy version
* Update pre-commit hooks (all #596)


0.12.8 (2022-11-16)
-------------------

* Fix look-ahead implementation in case of biased proposals (#568)

Minor:

* Remove boteh in test env as distributed #7227 got fixed
* Remove obsolete two gaussians test
* Fix Mixin random seed (set it via /dev/urandom)
* Update viserver to bokeh >= 3.0.1 (different import of TabPanel, Tabs)
  (all #589)
* Fix sqlalchemy warning "SAWarning: TypeDecorator BytesStorage()
  will not produce a cache key" (#590)


0.12.7 (2022-10-30)
-------------------

Minor:

* Ignore B027 empty method in an abstract base class
* Refactor EPSMixin
* Fix EPSMixin for ConcurrentFutureSampler
* Temporarily add bokeh to test due to dask error


0.12.6 (2022-08-30)
-------------------

Minor:

* Add JOSS paper of version 0.12.5
* Update Julia reference DiffEqJump -> JumpProcesses (name change)
* Unfix jinja2
* Update flake8 dependencies after new issues


0.12.5 (2022-06-21)
-------------------

Minor:

* Document outdated Google Colab version (Python 3.7)


0.12.4 (2022-05-05)
-------------------

Minor:

* Move near-zero population weight to warning (#563)
* Improve test installation instructions;
  fix for updates of flake8-print and black (#567)


0.12.3 (2022-04-05)
-------------------

* Document custom priors (#559)


0.12.2 (2022-03-25)
-------------------

* Update citation hints, add logo license (#554)


0.12.1 (2022-03-02)
-------------------

* Fix double logging in grid search


0.12.0 (2022-02-23)
-------------------

Major changes compared to 0.11.0:

New features:

* Add Silk acceptance rate curve based optimal threshold scheme (#539)
* Interface Copasi model simulators via BasiCO (thanks to Frank Bergmann)
  (#531)
* Interface simulators in the Julia language via pyjulia (#514)
* Add Wasserstein and Sliced Wasserstein optimal transport distances (#500)
* Finalize sensitivity weighted distance functions using inverse
  regression models and augmented regression targets (#478, 484)

Technical changes:

* Support python>=3.8 (#543)

Internals:

* Automatic code formatting via black, isort, and nbqa (#506, #508, #544)

Random:

* The logo is pink (#549)


0.11 series
...........


0.11.12 (2022-02-19)
--------------------

* Add Silk acceptance rate curve based optimal threshold scheme (#539)
* Fix Julia version in tests (#542)
* Apply nbqa black and isort to notebooks (#544)
* Update to stable tox
* Require python>=3.8 (#543)


0.11.11 (2021-12-25)
--------------------

* Implement color grouping by summary statistics and parameters in sankey
  plots (#536)


0.11.10 (2021-12-24)
--------------------

* Add BasiCO-Copasi model interface (thanks to Frank Bergmann) (#531)
* CI: Run Amici without sensitivity calculation (#533)


0.11.9 (2021-12-04)
-------------------

* Allow pickling of AMICI models (#527)
* Remove external handler cleanup as not used (#529)
* Allow timeout of external handlers (#530)


0.11.8 (2021-12-03)
-------------------

* Interface Julia simulators via pyjulia (#514)
* Refactor PCA distance, add tests (#518)
* Remove pyarrow as hard dependency for pandas storage (#523)
* Hierarchically structure examples, update "Parameter Inference"
  introduction (#524)
* Add minimum epsilon difference stopping condition (#525)


0.11.7 (2021-11-10)
-------------------

* Decompose ABCSMC.run for easier outer loop (#510)


0.11.6 (2021-11-05)
-------------------

* Unfix sphinx version for documentation (#509)
* Streamline function wrapper objects (#511)
* Remove rpy2 warning upon import of `pyabc.external` (#512)
* Move ot distance to scipy due to bug in pot 0.8.0 (#512)


0.11.5 (2021-10-29)
-------------------

* Regularly scheduled CI (#504)
* Fix Dask for Windows (#503)
* Apply the uncompromising Python code formatter black (#506)
* Apply automatic import sorting via isort (#508)


0.11.4 (2021-10-27)
-------------------

* Implement Wasserstein and Sliced Wasserstein distances (#500)
* Add env variable to bound population size in tests (#501)


0.11.3 (2021-10-16)
-------------------

* Update to amici 0.11.19 for scaled residual support (#491)
* Add links for online execution of notebooks on Google Colab and nbviewer
  (#492)
* Tests: Fix early stopping test for first generation (#494)


0.11.2 (2021-10-07)
-------------------

* Remove codacy due to excessive permission requests
* Tidy up example titles

0.11.1 (2021-10-06)
-------------------

Summary statistics:

* Allow transformed parameters as regression targets via `ParTrafo` (#478)
* Add Sankey flow plot (#484)
* Add "informative" notebook to document regression-based summary statistics
  and weights (#484)

Sampler:

* Speed up redis done-list checking by atomic operations (#482)


0.11.0 (2021-07-31)
-------------------

Diverse:

* Shorten date-time log (#456)
* Add look-ahead example notebook (#461)
* Fix decoration of `plot_acceptance_rates_trajectory` (#465)
* Hot-fix redis clean-up (#475)

Semi-automatic summary statistics and robust sample weighting (#429)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Breaking changes:

* API of the `(Adaptive)PNormDistance` was altered substantially to allow
  cutom definition of update indices.
* Internal weighting of samples (should not affect users).

Semi-automatic summary statistics:

* Implement (Adaptive)PNormDistance with the ability to learn summary
  statistics from simulations.
* Add `sumstat` submodule for generic mappings (id, trafos), and especially a
  `PredictorSumstat` summary statistic that can make use of `Predictor` objects.
* Add subsetting routines that allow restricting predictor model training
  samples.
* Add `predictor` submodule with generic `Predictor` class and concrete
  implementations including linear regression, Lasso, Gaussian Process,
  Neural Network.
* Add `InfoWeightedPNormDistance` that allows using predictor models to weight
  data not only by scale, but also by information content.

Outlier-robust adaptive distances:

* Update documentation towards robust distances.
* Add section in the corresponding notebook.
* Implement PCMAD outlier correction scheme.

Changes to internal sample weighting:

* Do not normalize weights of in-memory particles by model; this allows to
  more easily use the sampling weights and the list of particles for
  adaptive components (e.g. distance functions)
* Normalization of population to 1 is applied on sample level in the
  sampler wrapper function
* In the database, normalization is still by sample to not break old db
  support; would be nicer to also there only normalize by total sum
  -- requires a db update though.

Changes to internal object instruction from samples:

* Pass sample instead of weighted_sum_stats to distance function.
  This is because thus the distance can choose on its own what it wants
  -- all or only accepted particles; distances; weights; parameters;
  summary statistics.

Visualization:

* Function to plot adaptive distance weights from log file.


0.10 series
...........


0.10.16 (2021-05-11)
--------------------

* Allow color customization for `plot_credible_intervals` plots (#414)
* pyABC logo to grey to fit with both black and white backgrounds (#453)
* Add style set to global figure parameters, enabling dark mode (#454)


0.10.15 (2021-05-09)
--------------------

Sampler:

* Allow redis dynamical sampler to only wait for relevant particles after
  a generatio, giving a speed-up without drawbacks (#448)
* Add option to limit number of delayed look-ahead samples to limit memory
  usage (#428)

Logging:

* Standardize output of floats (#450)
* Use hierarchical logging (ABC.Submodule) (#417)

General:

* Refactor: Remove deprecated `nr_samples_per_parameter`, internal
  simplifications (#422)
* Tidy up and minimize dependencies (#436, #441)
* External: Remove simulation files after collecting results (#434)
* Make feather/pyarrow dependency optional for older hardware (#442)

Documentation:

* Add description of JupyterHub to documentation (#439)

CI:

* Test webserver basic functionality
* Rerun stochastically failing tests (all #436)
* Test whether dataframe storage routines work properly (#442)


0.10.14 (2021-02-21)
--------------------

General:

* Speed up parameter handling (30% internal speed-up) (#387)
* Streamline testing via tox and pre-commit hooks, add flake8 config file (#408)
* Update to python 3.9 (#411)
* Extract PEtab bounds and nominal parameters (#416)
* Allow specifying parameter names in density plots (#416)
* Normalize look-ahead sampling by subpopulation ESS (#418, #421)

Documentation:

* Update contribution and issue guidelines (#408)
* Add example on yaml2sbml usage (#416)
* Clean up user docs on parallelization and storage (#419)

Fixes:

* Fix redis for later started workers (#410)
* Fix PEtab parameter scale import, support all PEtab prior distributions (#413)

Database:

* Add database migration scheme based on alembic (#419)
* Store proposal ids, increment database version to 0 (#419)


0.10.13 (2021-02-04)
--------------------

* Update branch name master -> main in agreement with
  https://github.com/github/renaming (#406).
* Notebook improvements:

  * Add ground truth to noise notebook.
  * Rename notebook "quickstart" -> "model_selection".
  * Split notebook tests in 2, extend, reduce test matrix (python 3.7).
* Improve output at the beginning and end (e.g. no `end_time` at beginning;
  show duration).
* Add walltime plots (`plot_total_walltime`, `plot_walltime`,
  `plot_eps_walltime`).
* Make sure `ABCSMC.run()` is always properly finished (sampler, history)
  by a wrapper (all #401).
* Redis sampler with look-ahead mode:

  * Fix insufficient logging of look-ahead samples.
  * Log all accepted particles.
* Add `plot_lookahead_...` plots for look-ahead mode diagnostics.
* Add global figure parameter settings for pyABC (all #405).


0.10.12 (2021-01-20)
--------------------

* Check components for their adaptivity for correct application of look-ahead
  mode (#397).


0.10.11 (2021-01-02)
--------------------

* Require pandas >= 1.2.0 for PEtab.


0.10.10 (2021-01-01)
--------------------

* Uniquely identify redis runs via a uuid.
* Secure proper termination of processes for redis and dask (all #338).
* Rework redis sampler, implement a look-ahead mode for pre-defined (#338)
  and adaptive algorithms (#374) for near-perfect parallel efficiency.
* Allow subprocessing in multiprocessed redis workers (#377).
* Add a static-scheduling redis sampler (#379).
* Check whether epsilon is nan before sampling (#382).
* Remove useless IdentityFakeDistance (#390).


0.10.9 (2020-11-28)
-------------------

* Add stopping criterion for total walltime (#370).


0.10.8 (2020-11-27)
-------------------

* Allow to normalize acceptance rate plots by ESS (#346).
* Add a list of pyABC references (#348).
* Update to petabtests 0.0.0a5 (#362).
* Add stopping criterion for total number of samples (#364).
* Remove dill dependency, thus fixing a cloudpickle error, and
  run selected tests also with python 3.7 (#367).


0.10.7 (2020-08-20)
-------------------

* Move progress usage to attribute level (#336).
* Add check for whether redis is up already (#337).
* Add a self-tuned finite-space discrete transition (#341).


0.10.6 (2020-08-04)
-------------------

* Refactor and modularize ABCSMC inference module (#333).
* Make fast random choice function robust across dimensions (#333).


0.10.5 (2020-08-01)
-------------------

* Remove bkcharts dependency (#328).
* Add optional progress bar to various samplers (#330).
* Refactor package metadata (#330).
* Refactor CI build, add code quality tests (#331).
* Add warning when many zero-prior samples are generated (#331).


0.10.4 (2020-06-15)
-------------------

* Refactor `__all__` imports and docs API build (#312).
* Fix json export of aggregated adaptive distances (#316).
* Apply additional flake8 checks on code quality (#317).
* Assert model input is of type `pyabc.Parameter` (#318).
* Extend noise notebook to estimated noise parameters (#319).
* Implement optional pickling for multicore samplers; add MacOS
  pipeline tests (#320).


0.10.3 (2020-05-17)
-------------------

* Speed up multivariate normal multiple sampling (#299).
* Set default value for OMP_NUM_THREADS=1, stops warnings (#299).
* Base default number of parallel cores on PYABC_NUM_PROCS (#309).
* Update all notebooks to the latest numpy/scipy (#310).


0.10.2 (2020-05-09)
-------------------

* Update CI test system: latest Ubuntu, python 3.8, simplify R build (#296).
* Add weights logging to adaptive distances (#295).
* Migrate CI tests to GitHub Actions for speed-up, reliability and
  maintainability (#297, #298).


0.10.1 (2020-03-17)
-------------------

* Allow separate calibration population sizes, slightly reformulate
  PopulationStrategy class (#278).
* Allow specifying initial weights for adaptive distances, then without
  sampling from the prior (#279).
* Check PEtab test suite in tests (#281).


0.10.0 (2020-02-20)
-------------------

* Exact inference via stochastic acceptor finalized and tested (developed
  throughout the 0.9 series).
* Support basic PEtab functionality using AMICI ODE simulations (#268).
* Various error fixes (#265, #267).
* Log number of processes used by multiprocessing samplers (#263).
* Implement pyabc.acceptor.ScaledPDFNorm (#269).
* Implement list population size (#274, #276).
* On history loading, automatically find an id of a successful run (#273).


0.9 series
..........


0.9.26 (2020-01-24)
-------------------

* Add optional check whether database is non-existent, to detect typos.
* Set lower bound in 1-dim KDEs to <= 0 to not wrongly display near-uniform
  distributions. (both #257)
* Implement redis password protection for sampler and manage routine (#256).
* Make samplers available in global namespace (#249).
* Implement ListTemperature (#248).
* Allow plotting the relative ESS (#245).
* Allow resampling of weighted particles (#244).
* Fix ABCSMC.load with rpy2 (#242).


0.9.25 (2020-01-08)
-------------------

* Add summary statistics callback plot function (#231).
* Add possibility to log employed norms in StochasticAcceptor (#231) and
  temperature proposals in Temperature (#232).
* Implement optional early stopping in the MulticoreEvalParallelSampler and
  the SingleCoreSampler, when a maximum simulation number is exceeded
  (default behavior untouched).
* Log stopping reason in ABCSMC.run (all #236).
* Implement Poisson (#237) and negative binomial (#239) stochastic kernels.
* Enable password protection for Redis sampler (#238).
* Fix scipy deprecations (#234, #241).


0.9.24 (2019-11-19)
-------------------

* In ABCSMC.run, allow a default infinite number of iterations, and log the
  ESS in each iteration.
* Reformulate exponential temperature decay, allowing for a fixed number of
  iterations or fixed ratios.
* Solve acceptance rate temperature match in log space for numeric stability.
* Perform temperation of likelihood ratio in log space for numeric stability
  (all #221).
* Fix wrong maximum density value in binomial kernel.
* Allow not fixing the final temperature to 1 (all #223).
* Allow passing id to history directly (#225).
* Pass additional arguments to Acceptor.update.
* Give optional min_rate argument to AcceptanceRateScheme (all #226).
* In plot functions, add parameter specifying the reference value color (#227).


0.9.23 (2019-11-10)
-------------------

* Fix extras_require directive.
* Fix error with histogram plot arguments.
* Extend test coverage for visualization (all #215).
* ABCSMC.{new,load,run} all return the history with set id for convenience.
* Document pickling paradigm of ABCSMC class (see doc/sampler.rst).
* Always use lazy evaluation in updates (all #216).
* Restructure run function of ABCSMC class (#216, #218).
* Run notebooks on travis only on pull requests (#217).
* Correct weighting in AcceptanceRateScheme (#219).


0.9.22 (2019-11-05)
-------------------

* Fix error that prevented using rpy2 based summary statistics with non rpy2
  based models (#213).


0.9.21 (2019-11-05)
-------------------

* Introduce acceptor.StochasticAcceptor to encode the stochastic acceptance
  step generalizing the standard uniform criterion.
* Introduce distance.StochasticKernel to encode noise distributions, with
  several concrete implementations already.
* Introduce epsilon.Temperature to capture the temperature replacing the
  traditional epsilons. In addition, multiple concrete
  pyabc.epsilon.TemperatureSchemes have been implemented that handle the
  calculation of the next temperature value (all #197).


0.9.20 (2019-10-30)
-------------------

* Add high-level versions of the kde plotting routines (#204).
* Add unit tests for common epsilon schemes (#207).


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
