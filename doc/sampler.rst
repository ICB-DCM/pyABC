Parallel and Distributed Sampling Strategies
============================================

The pyABC package offers a variety of different parallel and distributed
sampling strategies. Single-core, multi-core and distributed execution is
supported in a couple different ways.
The ParticleParallel samplers implement the
"Static Scheduling (STAT)" strategy. The EvalParallel samplers implement the
"DynamicScheduling (DYN)" strategy.

Single-core execution
---------------------

For single-core execution, pyABC offers actually only a single sampler class,
namely the :class:`pyabc.sampler.SingleCoreSampler`.
This one just generates sample by sample sequentially.
It is, e.g. suitable for debugging purposes.


Multi-core only samplers
------------------------


For multi-core execution, pyABC comes with a number of possible parallelization
strategies.

There are two very specialized, multi-core only samplers.
First, the :class:`pyabc.sampler.MulticoreParticleParallelSampler` can be used
and implements the particle parallel sampling strategy.
Next, the :class:`pyabc.sampler.MulticoreEvalParallelSampler` implements the
evaluation parallel strategy.
Both sampler have very little communication overhead.
Moreover, they do not require to pickle the sample, evalualte and accept
functions.


Distributed-only samplers
-------------------------
Both distributed samplers implement the evaluation parallel strategy EPS.

The :class:`pyabc.sampler.RedisEvalParallelSampler` has low communication
overhead, and when running workers and redis-server locally is actually
competitive with the mutli-core only samplers. The
:class:`pyabc.sampler.RedisEvalParallelSampler` performs parameter sampling on
a per worker basis, and can handle fast function evaluations below 100ms
efficiently.

The :class:`pyabc.sampler.DaskDistributedSampler` has slightly higher
communication overhead, however this can be compensated with batch submission
mode. As the :class:`pyabc.sampler.DaskDistributedSampler` performs parameter
sampling locally on the master, it is unsuitable for simulation functions with
a runtime below 100ms, as network communication becomes prohibitive at this
point.

In general, the Redis based sampler will require slightly more effort in
setting up than the Dask based sampler, but has fewer constraints regarding
simulation function runtime. The Dask sampler is in turn better suited to
handle worker failures and unexpected execution host terminations.



Mutli-core and distributed samplers
-----------------------------------

Moreover, there are two more generic samplers which can be used in a
multicore and distributed fashion:

The :class:`pyabc.sampler.MappingSampler` can be used in a multi-core context
if the provided map implementation is a multi-core one, such as, e.g.
multiprocessing.Pool.map, or distributed if the map is a distributed one, such
as :class:`pyabc.sge.SGE.map`.

Similarly, the :class:`pyabc.sampler.ConcurrentFutureSampler` can use any
implementation of the python concurrent.futures.Executor interface. Again,
implementations are available for both multi-core (e.g.
concurrent.futures.ProcessPoolExecutor) and distributed (e.g. Dask)
environments

Check the :doc:`API documentation <sampler_api>` for more details.