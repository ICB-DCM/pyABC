.. _examples:

Examples
========

We provide a collection of example notebooks to get a better idea of how to
use pyABC, and illustrate core features.

The notebooks can be run locally with an installation of jupyter
(``pip install jupyter``), or online on Google Colab or nbviewer, following the
links at the top of each notebook.
To run the notebooks online, at least an installation of pyABC is required,
which can be performed by

.. code:: sh

   # install if not done yet
   !pip install pyabc --quiet

Potentially, further dependencies may be required.

Getting started
---------------

.. toctree::
   :maxdepth: 2

   examples/parameter_inference.ipynb
   examples/model_selection.ipynb

Algorithms and features
-----------------------

.. toctree::
   :maxdepth: 2

   examples/early_stopping.ipynb
   examples/resuming.ipynb
   examples/adaptive_distances.ipynb
   examples/informative.ipynb
   examples/aggregated_distances.ipynb
   examples/wasserstein.ipynb
   examples/data_plots.ipynb
   examples/noise.ipynb
   examples/discrete_parameters.ipynb
   examples/look_ahead.ipynb

Languages
---------

.. toctree::
   :maxdepth: 2

   examples/using_R.ipynb
   examples/using_julia.ipynb
   examples/external_simulators.ipynb
   examples/petab_yaml2sbml.ipynb

Application examples
--------------------

.. toctree::
   :maxdepth: 2

   examples/conversion_reaction.ipynb
   examples/chemical_reaction.ipynb
   examples/multiscale_agent_based.ipynb
   examples/sde_ion_channels.ipynb
   examples/petab_application.ipynb

Download the examples as notebooks
----------------------------------

* :download:`Parameter inference <examples/parameter_inference.ipynb>`
* :download:`Model selection <examples/model_selection.ipynb>`
* :download:`Early stopping of model simulations <examples/early_stopping.ipynb>`
* :download:`Resuming stored ABC runs <examples/resuming.ipynb>`
* :download:`Using R with pyABC <examples/using_R.ipynb>`
* :download:`Using Julia with pyABC <examples/using_julia.ipynb>`
* :download:`Ordinary differential equations: Conversion reaction <examples/conversion_reaction.ipynb>`
* :download:`Markov jump process: Reaction network <examples/chemical_reaction.ipynb>`
* :download:`Multi-scale model: Tumor spheroid growth <examples/multiscale_agent_based.ipynb>`
* :download:`Stochastic Differential Equation: Ion channel noise in Hodgkin-Huxley neurons <examples/sde_ion_channels.ipynb>`
* :download:`Adaptive distances <examples/adaptive_distances.ipynb>`
* :download:`Informative distances and summary statistics <examples/informative.ipynb>`
* :download:`Aggregated distances <examples/aggregated_distances.ipynb>`
* :download:`Wasserstein distances <examples/wasserstein.ipynb>`
* :download:`External simulators <examples/external_simulators.ipynb>`
* :download:`Data plots <examples/data_plots.ipynb>`
* :download:`Measurement noise and exact inference <examples/noise.ipynb>`
* :download:`PEtab import and yaml2sbml <examples/petab_yaml2sbml.ipynb>`
* :download:`PEtab application example <examples/petab_application.ipynb>`
* :download:`Discrete parameters <examples/discrete_parameters.ipynb>`
* :download:`Look-ahead sampling <examples/look_ahead.ipynb>`


.. warning::

   Upgrade to the latest pyABC version before running the examples.
   If you installed pyABC some weeks (or days) a ago, some new features might
   have been added in the meantime.
   Refer to the :ref:`upgrading` section on how to upgrade pyABC.
