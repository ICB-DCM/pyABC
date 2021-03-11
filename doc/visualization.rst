.. _web_visualization:

Visualization and analysis
==========================

Overview
--------

pyABC offers a variety of routines to visualize and analyze results.
Usage examples are given in the :ref:`Examples section<examples>`.
An overview of all plotting routines can be found in the
:ref:`Visualization API documentation <api_visualization>`.
Further, functions to correctly extract statistics, in particular
on importance samples obtained via ABCSMC, can be found in the
:ref:`Weighted statistics API documentation <api_weightedstatistics>`.


Web-based visualizations
------------------------

Further, pyABC comes with a web server, which displays lots of useful
information on the currently running and already completed ABC tasks.
You can launch it from the command line with

.. code-block:: bash

    abc-server <databasename>


It opens per default a web server on port 5000.

You should see something similar to the following:

.. image:: server_screenshots/main.png


Via "Go to ABC Run List", you can see all running and finished ABC runs, which you can then inspect in more detail.

You can get overviews over the models:

.. image:: server_screenshots/model_overview.png

Information about individual model parameters for each model and time point is also displayed:

.. image:: server_screenshots/model_detail.png



Type in the command line

.. code-block:: bash

   abc-server --help

To get more information on available options, such as selecting another port:


.. code-block:: bash

   abc-server --port=8888 <databasename>
