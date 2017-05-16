Parallel and Distributed Sampling Strategies
============================================

The pyABC package offers a variety of different parallel and distributed
sampling strategies. Single-core, multi-core and distributed execution is
supported in a couple different ways.
The ParticleParallel samplers and the MappingSampler implement the
"Static Scheduling (STAT)" strategy. The EvalParallel samplers,
the DaskDistributedSampler and the ConcurrentFutureSampler implement the
"DynamicScheduling (DYN)" strategy.
The batchsize argument of the DaskDistributedSampler and the
ConcurrentFutureSampler allow to interpolate between dynamic and static
scheduling.


Single-core execution
---------------------

For single-core execution, pyABC offers
the :class:`pyabc.sampler.SingleCoreSampler`.
This one just generates sample by sample sequentially.
This sampler is intended for debugging purposes as debugging parallel
code can be hard sometimes.


Multi-core only samplers
------------------------


For multi-core execution, pyABC implements two possible parallelization
strategies.

* First, the :class:`pyabc.sampler.MulticoreParticleParallelSampler`
  implements the STAT sampling strategy.
* Next, the :class:`pyabc.sampler.MulticoreEvalParallelSampler` implements the
  DYN strategy. This is currently the default sampler.
Both samplers are highly specialized to the multi-core setting and
have very little communication overhead.
Even for very small model evaluation times
these samplers are about as fast as the single core sampler.
This is achieved circumventing object serialization by forking.
As Microsoft Windows does not support forking, these samplers might not
work as expected on Windows.


Distributed samplers
--------------------

The distributed samplers can, be used in a distributed setting, and of course
also locally by setting up a local cluster. However, for local execution,
the multi-core samplers are recommended as they are easier to set up.


The :class:`pyabc.sampler.RedisEvalParallelSampler` has very low communication
overhead, and when running workers and redis-server locally is actually
competitive with the mutli-core only samplers.
The :class:`pyabc.sampler.RedisEvalParallelSampler`
performs parameter sampling on a per worker basis, and can handle fast
function evaluations efficiently.

The :class:`pyabc.sampler.DaskDistributedSampler` has slightly higher
communication overhead, however this can be compensated with the batch
submission mode. As the :class:`pyabc.sampler.DaskDistributedSampler`
performs parameter sampling locally on the master,
it is unsuitable for simulation functions with a runtime below 100ms,
as network communication becomes prohibitive at this point.


The Redis based sampler cab require slightly more effort in
setting up than the Dask based sampler, but has fewer constraints regarding
simulation function runtime. The Dask sampler is in turn better suited to
handle worker failures and unexpected execution host terminations.



Gemeral extensible samplers
---------------------------

Moreover, there are two more generic samplers which can be used in a
multicore and distributed fashion.
These samplers facilitate adaptation of pyABC to new parallel environments.


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