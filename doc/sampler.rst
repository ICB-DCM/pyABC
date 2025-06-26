.. _sampler:


Parallel sampling
=================

pyABC offers a variety of multi-core parallel and distributed samplers,
which handle the usually most time-expensive part of an ABC analysis:
the simulation of data from the model for sampled parameters, the
generation of summary statistics, and the calculation of the distance
of simulated and observed data.

The most-used and best-supported samplers are the
:class:`pyabc.sampler.MulticoreEvalParallelSampler` for multi-processed
sampling, the
:class:`pyabc.sampler.RedisEvalParallelSampler` for distributed sampling,
and for deterministic sampling purposes the non-parallelized
:class:`pyabc.sampler.SingleCoreSampler`. These should be preferably used,
however also the other parallelization engines mentioned below should work.


Strategies
----------

The various samplers implement two different sampling strategies:
"Static Scheduling (STAT)" and "Dynamic Scheduling (DYN)". STAT minimizes
the total execution time, whereas DYN minimizes the wall-time and is
generally preferable as it finishes faster. For details see [Klinger2018]_.

The `ParticleParallel` samplers, the `MappingSampler` and the
`RedisStaticSampler` implement the "Static Scheduling (STAT)" strategy.

The `EvalParallel` samplers, the `DaskDistributedSampler` and the
`ConcurrentFutureSampler` implement the "Dynamic Scheduling (DYN)"
strategy.

The batchsize arguments of the `DaskDistributedSampler`, the
`ConcurrentFutureSampler` and the `RedisEvalParallelSampler`
allow to interpolate between dynamic and static scheduling and to reduce
communication overhead.

.. [Klinger2018] Emmanuel Klinger, Dennis Rickert, Jan Hasenauer.
   pyABC: distributed, likelihood-free inference.
   Bioinformatics 2018; https://doi.org/10.1093/bioinformatics/bty361


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
competitive with the multi-core only samplers.
The :class:`pyabc.sampler.RedisEvalParallelSampler`
performs parameter sampling on a per worker basis, and can handle fast
function evaluations efficiently. Further, it is the only sampler that
allows proactive sampling to minimize the overall wall-time
("look-ahead mode").
The :class:`pyabc.sampler.RedisStaticSampler`
implements static scheduling and may be of interest if the simulation time
needs to be minimized.

The :class:`pyabc.sampler.DaskDistributedSampler` has slightly higher
communication overhead, however this can be compensated with the batch
submission mode. As it performs parameter sampling locally on the master,
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

Check the :ref:`API documentation <api_sampler>` for more details.


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
function are defined, configure pyABC to use the redis sampler. For the
:class:`pyabc.sampler.RedisEvalParallelSampler`, use

.. code:: python

   from pyabc.sampler import RedisEvalParallelSampler

   redis_sampler = RedisEvalParallelSampler(host="111.111.111.111", port=6379)

   abc = pyabc.ABCSMC(models, priors, distance, sampler=redis_sampler)

If password protection is used, in addition pass the argument
``password=PASSWORD`` to the RedisEvalParallelSampler.

For the :class:`pyabc.sampler.RedisStaticSampler`, the same applies, no
modifications of the workers are necessary.

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

pyABC has been successfully employed on various high-performance computing
(HPC) infrastructures. There are a few things to keep in mind.


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

When submitting a large number of individual SLURM jobs (``n_jobs``), the
scheduler could be overloaded, i.e. increased scheduling overhead may degrade
the overall effiency of the scheduling on the HPC system.

As an alternative, consider to use SLURM job arrays. A SLURM job array is a feature
to manage a collection of similar jobs efficiently using a single submission script.
Each job in the array (task), shares the same job script but can operate on
different inputs and parameters identified by an unique index ``$SLURM_ARRAY_TASK_ID``.

Furthermore, monitoring and job control is streamlined compared to numerous individual
jobs scattered across the queue (scalability of job submission). SLURM is optimized to handle
large job arrays efficienctly and should be thus considered as an alternative to to the
submission of many individual, yet related or similar jobs.


.. code:: bash

   sbatch --array=0-99 script_redis_worker script_redis_worker.sh ${SLURM_ARRAY_TASK_ID}

Using ``--array`` one specifies the number of jobs (here ``n_jobs`` is manually set to 99, resulting in 100 tasks)
and note that depending on the variable ``${SLURM_ARRAY_TASK_ID}`` the script ``script_redis_worker.sh`` could 
handle for instance different input parameters or input files as identified by a unique index.


Note that when planning for the number of overall redis workers, batches, and
cpus per batch, also the parallelization on the level of the simulations has
to be taken into account. Also, memory requirements should be checked in
advance.


JupyterHub
~~~~~~~~~~

As an intermediate step between local desktop systems and a full HPC cluster
system, Jupyterhub (https://jupyterhub.readthedocs.io/en/stable/) can speed up
computations with pyABC without requiring detailed knowledge about the HPC
system.
From the user’s perspective, JupyterHub provides a web based interface to
computational resources through standard Jupyter notebooks.
With minimal adaptions to the local workflow, the resources of one full
computing node of a cluster can be utilized for pyABC by using the multicore
samplers.
Depending on the hardware, this could be as much as 128 CPU cores.
No interaction with the command line and the batch system is required.
Jupyterhub is installed on many HPC centers, e.g. at the Centre for Information
Services and High Performance Computing at TU Dresden
(https://doc.zih.tu-dresden.de/hpc-wiki/bin/view/Compendium/JupyterHub)
or at the Juelich Supercomputing Centre (https://jupyter-jsc.fz-juelich.de).


Pickling
--------

.. note::

   This section is of interest to developers, or if you encounter memory
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


SGE cluster scheduling
----------------------

Quick start
~~~~~~~~~~~

The pyabc.sge package provides as most important class
the :class:`SGE <pyabc.sge.SGE>`. Its ``map`` method
automatically parallelizes across an SGE/UGE cluster.
The SGE class can be used in standalone mode or in combination
with the ABCSMC class (see below :ref:`sge-abcsmc`).

Usage of the parallel package is fairly easy. For example:

.. code-block:: python

   from pyabc.sge import SGE
   sge = SGE(priority=-200, memory="3G")

   def f(x):
       return x * 2

   tasks = [1, 2, 3, 4]

   result = sge.map(f, tasks)

   print(result)


.. parsed-literal::

   [2, 4, 6, 8]


The job scheduling is either done via an SQLite database or a REDIS instance.
REDIS is recommended as it works more robustly, in particular in cases
where distributed file systems are rather slow.

.. note::

   A configuration file in ``~/.parallel`` is required.
   See :class:`SGE <pyabc.sge.SGE>`.

The ``pyabc.sge.sge_available`` can be used to check if an SGE cluster can be used on the machine.

Check the :ref:`API documentation <api_sge>` for more details.


Information about running jobs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the ``python -m pyabc.sge.job_info_redis`` to get a nicely formatted output
of the current execution state, in case the REDIS mode is used.
Check ``python -m pyabc.sge.job_info_redis --help`` for more details.


.. _sge-abcsmc:

Usage notes
~~~~~~~~~~~

The :class:`SGE <pyabc.sge.SGE>` class can be used in standalone mode for
convenient parallelization of jobs across a cluster, completely independent
of the rest of the pyABC package.
The :class:`SGE <pyabc.sge.SGE>` class can also be combined, for instance, with
the :class:`pyabc.sampler.MappingSampler` class for simple parallelization
of ABC-SCM runs across an SGE cluster.
