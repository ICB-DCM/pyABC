Welcome to pyABC's documentation!
=================================

.. image:: https://travis-ci.org/neuralyzer/pyabc.svg?branch=master
    :target: https://travis-ci.org/neuralyzer/pyabc

Version: |version|


pyABC is a framework for distributed, likelihood-free inference.
That means, if you have a model and some data and want to know the posterior
distribution over the model parameters, i.e. you want to know with which
probability which parameters explain the observed data, then pyABC might be
for you.

All you need is some way to numerically draw samples from the model, given
the model parameters. pyABC "inverts" the model for you and tells you
which parameters were well matching and which ones not. You do **not** need
to analytically calculate the likelihood function.

pyABC runs efficiently on multi-core machines and distributed cluster setups.
It is easy to use and flexibly extensible.

If you're interested in using it, you can cite the preprint available here:
https://doi.org/10.1101/162552

User's Guide
------------

This part of the documentaiton guides you step by step through
the usage of this package.

.. toctree::
   :maxdepth: 2

   what
   examples/quickstart.ipynb
   installation
   web_visualization
   sampler
   sge
   examples
   export_db
   releasenotes
   license



API reference
-------------

This part of the documentation describes the individual classes, functions,
modules and packages.

.. toctree::
   :maxdepth: 2

   abcsmc_api
   distance_api
   model_api
   epsilon_api
   datastore_api
   transition_api
   populationstrategy_api
   sampler_api
   parameters_api
   random_variables_api
   sge_api
   external_API
   visualization_api


Developer's Guide
-----------------

Interested in contributing? It is not that hard.

.. toctree::
   :maxdepth: 2

   documentation
   code



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. todolist::
