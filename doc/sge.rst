Parallel job execution on an SGE cluster environment
====================================================

Quick start
-----------

The parallel package provides as main class the :class:`SGE <pyabc.sge.SGE>`. It's ``map`` method
automatically parallelizes across an SGE cluster.

Usage of the parallel package is farly easy. For example

.. code::

   from parallel import SGE
   sge = SGE(priority=-200, memory="3G")

   def f(x):
       return x * 2

   tasks = [1, 2, 3, 4]

   result = sge.map(f, tasks)

   print(result)

.. parsed-literal::
   [2, 4, 6, 8]


The job scheduling is either done via a SQLite database or a REDIS instance. REDIS is recommended as it works
more robustly.

.. note::

   A configuration file in ``~/.parallel`` is required.
   See :class:`SGE <pyabc.sge.SGE>`.


The ``pyabc.sge.sge_available`` can be used to check if an SGE cluster can be used on the machine.

Check the :doc:`API documentation <sge_api>` for more details.


Information about running jobs
------------------------------

Use the ``python -m pyabc.sge.job_info_redis`` to get a nicely formatted output of the current execution state.
Check ``python -m pyabc.sge.job_info_redis --help`` for more details.
