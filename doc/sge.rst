.. _sge:

Parallel job execution on an SGE cluster environment
====================================================

Quick start
-----------

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

Check the :doc:`API documentation <api_sge>` for more details.


Information about running jobs
------------------------------

Use the ``python -m pyabc.sge.job_info_redis`` to get a nicely formatted output
of the current execution state, in case the REDIS mode is used.
Check ``python -m pyabc.sge.job_info_redis --help`` for more details.


.. _sge-abcsmc:

Usage notes
-----------

The :class:`SGE <pyabc.sge.SGE>` class can be used in standalone mode for
convenient parallelization of jobs across a cluster, completely independent
of the rest of the pyABC package.
The :class:`SGE <pyabc.sge.SGE>` class can also be combined, for instance, with
the :class:`pyabc.sampler.MappingSampler` class for simple parallelization
of ABC-SCM runs across an SGE cluster.
