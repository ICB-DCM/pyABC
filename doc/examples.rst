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
   examples/optimal_threshold.ipynb
   examples/discrete_parameters.ipynb
   examples/look_ahead.ipynb

External interfaces
-------------------

.. toctree::
   :maxdepth: 2

   examples/using_R.ipynb
   examples/using_julia.ipynb
   examples/external_simulators.ipynb
   examples/petab_yaml2sbml.ipynb
   examples/using_copasi.ipynb

Application examples
--------------------

.. toctree::
   :maxdepth: 2

   examples/conversion_reaction.ipynb
   examples/chemical_reaction.ipynb
   examples/multiscale_agent_based.ipynb
   examples/sde_ion_channels.ipynb
   examples/petab_application.ipynb

.. warning::

   Upgrade to the latest pyABC version before running the examples.
   If you installed pyABC some weeks (or days) a ago, some new features might
   have been added in the meantime.
   Refer to the :ref:`upgrading` section on how to upgrade pyABC.
