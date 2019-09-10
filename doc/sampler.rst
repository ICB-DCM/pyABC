.. _sampler:


Parallel and Distributed Sampling
=================================


Strategies
----------

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
~~~~~~~~~~~~~~~~~~~~~

For single-core execution, pyABC offers
the :class:`pyabc.sampler.SingleCoreSampler`.
This one just generates sample by sample sequentially.
This sampler is intended for debugging purposes as debugging parallel
code can be hard sometimes.


Multi-core only samplers
~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~

The distributed samplers can be used in a distributed setting, and of course
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


The Redis based sampler can require slightly more effort in
setting up than the Dask based sampler, but has fewer constraints regarding
simulation function runtime. The Dask sampler is in turn better suited to
handle worker failures and unexpected execution host terminations.


General extensible samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Check the :doc:`API documentation <api_sampler>` for more details.


How to setup a Redis based distributed cluster
----------------------------------------------


Step 1: Start a Redis server without password authentication
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Start one some machine, which is reachable by the machine running the pyABC
main application and by the workers, a Redis server, disabling password
authentication:


.. code:: bash

   redis-server --protected-mode no


You should get an output looking similar to the one below:

.. literalinclude:: redis_setup/redis_start_output.txt
   :language: bash

If you're on Linux, you can install redis either via your package manager
of if you're using anaconda via

.. code:: bash

   conda install redis

At this point, Windows is not officially supported by the Redis developers.
We assume for now, that the IP address of the machine running the Redis server
is 111.111.111.111.


Step 2 or 3: Start pyABC
~~~~~~~~~~~~~~~~~~~~~~~~

It does not matter what you do first: starting pyABC or starting the
workers. Assuming the models, priors and the distance function are defined,
configure pyABC to use the Redis sampler

.. code:: python

   from pyabc.sampler import RedisEvalParallelSampler

   redis_sampler = RedisEvalParallelSampler(host="111.111.111.111")

   abc = pyabc.ABCSMC(models, priors, distance, sampler=redis_sampler)

Note that 111.111.111.111 is the IP address of the machine running the Redis
server. Then start the ABC-SMC run as usual with

.. code:: python

   abc.run(...)


passing the stopping conditions.


Step 2 or 3: Start the workers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It does not matter what you do first: starting pyABC or starting the
workers. You can even dynamically add workers after the sampling has started.
Start as many workers as you whish on the machines you whish. Up to 10,000
workers should not pose any problem if the model evaluation times are on the
second scale or longer.

.. code:: bash

    abc-redis-worker --host=111.111.111.111

Again, 111.111.111.111 is the IP address of the machine running the Redis
server. You should get an output similar to


.. code:: bash

   INFO:REDIS-WORKER:Start redis worker. Max run time 7200.0s, PID=2731

Note that the ``abc-redis-worker`` command also has options to set the
maximal runtime of a worker, e.g. ``--runtime=2h``, ``--runtime=3600s``,
``--runtime=2d``, to start a worker running for 2 hours, 3600 seconds
or 2 days.
The default is 2 hours. It is OK if a worker
stops during the sampling of a generation. You can add new workers during
the sampling process.
The ``abc-redis-worker`` command also has an option ``--processes`` which
allows you to start several worker procecces in parallel.
This might be handy in situations where you have to use a whole cluster node
with several cores.


Optional: Monitoring
~~~~~~~~~~~~~~~~~~~~

pyABC ships with a small utility to manage the Redis based sampler setup.
To monitor the ongoing sampling, execute


.. code:: bash

   abc-redis-manager info --host=111.111.111.111

again, assuming 111.111.111.111 is the IP of the Redis server. If no sampling
has happened yet, the output should look like

.. code:: bash

   Workers=None Evaluations=None Particles=None

The keys are to be read as follows:

* Workers: Currently sampling workers. This will show None or zero, even if
  workers are connected but they are not running. This number drops to zero
  at the end of a generation.
* Evaluations: Number of accumulated model evaluations for the current
  generations. This is a sum across all workers.
* Particles: Number of particles which remain to be accepted.
  This number decreases over the
  course of a population and reaches 0 (or a negative number due to excess
  sampling) at the end of a population. At the very start, this is just the
  population size.


Optional: Stopping workers
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``abc-redis-manager stop`` to send a signal to the workers that they should
shutdown at the end of the current generation.

You can also stop workers with ``Ctrl-C``, or even send a kill signal when
pyABC has finished.


Optional: Something with the workers went wrong in the middle of a run
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It can happen that workers get unexpectedly killed.
If they are not able to communicate to the redis-server that they've finished
working on the current population before they're killed,
the pyABC master process will wait forever.
In such cases, the following can be done

1. Terminate all running workers (but not the pyABC master process and also
   not the redis-server)
2. Execute ``abc-redis-manager reset-workers`` to manually reset the number
   of registered workers to zero.
3. Start worker processes again.
