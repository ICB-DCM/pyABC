Parallel and Distributed Sampling Strategies
============================================

The pyABC package offers a variety of different parallel and distributed
sampling strategies. Single-core, multi-core and distributed execution is
supported in a couple different ways.

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

The :class:`pyabc.sampler.RedisEvalParallelSampler` is intended for distributed
execution and implements the evaluation parallel strategy.
However, it has really very low overhad, and by running the workers and
the redis-server locally it is actually competetive with the mutli-core
only samplers

The :class:`pyabc.sampler.DaskDistributedSampler` really only makes sense
for distributed exectuion of long running simulation due to its substantial
communication overhead.
Even when the dask-scheduler is run locally and the workers as well, ths
communication overhead is still substantial.



Mutli-core and distributed samplers
-----------------------------------

Moreover, there are two more generic samplers which can be used in a
multicore and distributed fashion: The :class:`pyabc.sampler.MappingSampler` can
be used in a multi-core context if the provided map implementation is a
multi-core one, such as, e.g. multiprocessing.Pool.map, or distributed if the
map is a distributed one, such as :class:`pyabc.sge.SGE.map`.
Moreover, there is a sampler which supports the concurrent futures interface



Check the :doc:`API documentation <sampler_api>` for more details.