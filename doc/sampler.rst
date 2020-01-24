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


How to set up a Redis based distributed cluster
-----------------------------------------------


Step 0: Prepare the redis server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run the redis server, use a machine which is reachable both by the main
application and by the workers. If you're on Linux, you can install redis
either via your package manager, or, if you're using anaconda, via

.. code:: bash

   conda install redis

Windows is currently not officially supported by the redis developers.

It is recommended to run a redis server only with password protection, since
it otherwise accepts any incoming connection. To set up password protection
on the server, you need to modify the ``redis.conf`` file. Usually, such a file
exists under ``REDIS_INSTALL_DIR/etc/redis.conf``. You can however also set
up your own file. It suffices to add or uncomment the line

.. code:: bash
   
    requirepass PASSWORD

where `PASSWORD` should be replaced by a more secure password.


Step 1: Start a redis server
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this example, we assume that the IP address of the machine running the
redis server is ``111.111.111.111`` (the default is ``localhost``),
and that the server should listen on port ``6379`` (the redis default).

If password protection is used, start the server via

.. code:: bash

    redis-server /path/to/redis.conf --port 6379

If no password protection is required, instead use

.. code:: bash

    redis-server --port 6379 --protected-mode no

You should get an output looking similar to the one below:

.. literalinclude:: redis_setup/redis_start_output.txt
   :language: bash


Step 2 or 3: Start pyABC
~~~~~~~~~~~~~~~~~~~~~~~~

It does not matter what you do first: starting pyABC or starting the
workers. In your main program, assuming the models, priors and the distance
function are defined, configure pyABC to use the redis sampler

.. code:: python

   from pyabc.sampler import RedisEvalParallelSampler

   redis_sampler = RedisEvalParallelSampler(host="111.111.111.111", port=6379)

   abc = pyabc.ABCSMC(models, priors, distance, sampler=redis_sampler)

If password protection is used, in addition pass the argument
``password=PASSWORD`` to the RedisEvalParallelSampler.
   
Then start the ABC-SMC run as usual with

.. code:: python

   abc.run(...)


passing the stopping conditions.


Step 2 or 3: Start the workers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It does not matter what you do first: starting pyABC or starting the
workers. You can even dynamically add workers after the sampling has started.
Start as many workers as you wish on the machines you wish. Up to 10,000
workers should not pose any problem if the model evaluation times are on the
scale or seconds or longer. You start workers on your cluster via

.. code:: bash

    abc-redis-worker --host=111.111.111.111 --port=6379

If password protection is used, you need to append ``--password=PASSWORD``.
You should get an output similar to

.. code:: bash

   INFO:REDIS-WORKER:Start redis worker. Max run time 7200.0s, PID=2731

The ``abc-redis-worker`` command has further options (see them via
``abc-redis-worker --help``), in particular to set the
maximal runtime of a worker, e.g. ``--runtime=2h``, ``--runtime=3600s``,
``--runtime=2d``, to start a worker running for 2 hours, 3600 seconds
or 2 days (the default is 2 hours). It is OK if a worker
stops during the sampling of a generation. You can add new workers during
the sampling process at any time.
The ``abc-redis-worker`` command also has an option ``--processes`` which
allows you to start several worker procecces in parallel, e.g.
``--processes=12``. This might be handy in situations where you have to use
a whole cluster node with several cores.


Optional: Monitoring
~~~~~~~~~~~~~~~~~~~~

pyABC ships with a small utility to manage the Redis based sampler setup.
To monitor the ongoing sampling, execute


.. code:: bash

   abc-redis-manager info --host=111.111.111.111

If password protection is used, you need to specify the password via
``--password=PASSWORD``.
If no sampling has happened yet, the output should look like

.. code:: bash

   Workers=None Evaluations=None Acceptances=None/None

The keys are to be read as follows:

* Workers: Currently sampling workers. This will show None or zero, even if
  workers are connected but they are not running. This number drops to zero
  at the end of a generation.
* Evaluations: Number of accumulated model evaluations for the current
  generations. This is a sum across all workers.
* Acceptances: In ``i/n``, ``i`` particles out of a requested population
  size of ``n`` have been accepted already. It can be ``i>n`` at the end of
  a population due to excess sampling.


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


High-performance infrastructure
-------------------------------

pyABC has been successfully employed on various high-performance computing (HPC) infrastructures. There are a few things to keep in mind.


Long-running master process
~~~~~~~~~~~~~~~~~~~~~~~~~~~

While most of the work happens on parallel workers, pyABC requires one
long-running master process in the background for all of the analysis
(or rather two processes, namely the master process running the execution
script, and in addition possibly a task scheduler like the redis server).
If the HPC infrastructure does not allow for such long-running processes
with low CPU and memory requirements, one has to find a way around.
Eventually, it is planned for pyABC to support loss-free automatic
checkpointing and restarting, but presently this is not yet implemented.
If possible, the master process can be run on external servers, login nodes,
or on execution nodes while taking maximum runtimes and reliability of
server and connections into consideration.


Job scheduling
~~~~~~~~~~~~~~

HPC environments usually employ a job scheduler for distributing work to the
execution nodes. Here, we shortly outline how pyABC can be integrated in such
a setup. Exemplarily, we use a redis sampler, usage of in particular the dask
sampler being similar.

Let us consider the widely used job scheduler
`slurm <https://slurm.schedmd.com>`_. First, we need a script
``script_redis_worker.sh`` that starts the redis worker:

.. code:: bash

   #!/bin/bash
   
   # slurm settings
   #SBATCH -p {partition_id}
   #SBATCH -c {number_of_cpus}
   #SBATCH -t {time_in_minutes}
   #SBATCH -o {output_file_name}
   #SBATCH -e {error_file_name}

   # prepare environment, e.g. set path

   # run
   abs-redis-worker --host={host_ip} --port={port} --runtime={runtime} \
       --processes={n_processes}

Here, ``n_processes`` defines the number of processes started for that batch
job via multiprocessing. Some HPC setups prefer larger batch jobs, e.g. on a
node level, so here each job can already be given some parallelity. The
``SBATCH`` macros define the slurm setting to be used.

The above script would be submitted to the slurm job manager via ``sbatch``.
It makes sense to define a script for this as well:

.. code:: bash

   #!/bin/bash

   for i in {1..{n_jobs}}
   do
     sbatch script_redis_worker.sh
   done

Here, ``n_jobs`` would be the number of jobs submitted. When the job scheduler
is based on qsub, e.g. SGE/UGE, instead use a script like

.. code:: bash

   #!/bin/bash

   for i in {1..{n_jobs}}
   do
     qsub -o {output_file_name} -e {error_file_name} \
         script_redis_worker.sh

and adapt the worker script. For both, there exist many more configuration
options. For further details see the respective documentation.

Note that when planning for the number of overall redis workers, batches, and
cpus per batch, also the parallelization on the level of the simulations has
to be taken into account. Also, memory requirements should be checked in
advance.


Pickling
--------

.. note::

   This section is of interest to developers, of if you encounter memory
   problems.

For most of the samplers, pyABC uses
`cloudpickle <https://github.com/cloudpipe/cloudpickle>`_ to serialize objects
over the network and run simulations on remote nodes. In particular, this
enables us to use lambda functions.

However, care must be taken w.r.t. the size of the serialized object, i.e. to
only include what is really required. This is why in the `pyabc.ABCSMC` class
we had to write some functions that prevent the whole ABCSMC object from being
serialized. For developers, the following example illustrates the problem:

.. code-block:: python

   import cloudpickle as pickle
   import numpy as np

   class A:

      def __init__(self):
         self.big_arr = np.eye(10000)
         self.small_arr = np.zeros(2)

      def costly_function(self):
         def fun(x):
            print(self.small_arr, x)

         return fun

      def efficient_function(self):
         small_arr = self.small_arr

         def fun(x):
            print(small_arr, x)
         
         return fun


   a = A()

   print("The whole class:")
   print(len(pickle.dumps(a)))
   # 800001025

   print("Costly function:")
   print(len(pickle.dumps(a.costly_function())))
   # 800001087

   print("Efficient function:")
   print(len(pickle.dumps(a.efficient_function())))
   # 522
