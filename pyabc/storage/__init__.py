"""
Datastore
=========


Purpose of the datastore
------------------------

The most important class here is the History class. The History class
is the interface to the database in which pyABC stores and logs information
during the ABC-SMC run, but also the interface which allows you to query that
information later on.

Initializing the databse interface from a file
----------------------------------------------

For querying, you initialize a History object with a valid SQLAlchmey
database identifier. For example, if you ABC-SMC data is stored in a file
"data.db", you initialize the History with:

.. code-block:: python

   history = History("sqlite:///data.db")

Don't mind the three slashes. This is SQLAlchemy syntax.

If more than one ABC-SMC run is stored in your database file, these runs will
have IDs. The first run has ID=1, the second run ID=2 and so on. Per default,
the first run found in the database is automatically selected. To select a
specific run by manually, do

.. code-block:: python

    history.id = n

if n is the run number, e.g. n=3.


Querying the database
---------------------

The History class has a number of methods which are relevant for querying the
stored data. The most important ones are:

* ``History.get_distribution``
  to retrieve information on the parameter posteriors,
* ``History.get_model_probabilities`` to retrieve information on the model
  probabilities in case you're doing model selection,
* ``History.get_all_populations``,  to retrieve information on the evolution
  of the acceptance threshold and the number of sample attempts per population,
* ``History.get_nr_particles_per_population``, to retrieve the number of
  particles per population (this number os not necessariliy constant),
* ``History.get_weighted_distances``, to retrieve the distances the parameter
  samples achieved,
* ``History.n_populations`` to get the total number of populations, and
* ``History.total_nr_simulations`` to get the total number of simulations, i.e.
  sample attempts.

Use ``get_distribution`` to retrieve your posterior particle population. For
example,

.. code-block:: python

   df, w = history.get_distribution(m)

will return a DataFrame df of parameters and an array w of weights of the
particles of model m in the last available population.
If you're interested in intermediate
populations, add the optional t parameter, which indicates the population
number (the first population is t=0)

.. code-block:: python

   df, w = history.get_distribution(m, t)

"""

from .history import History

__all__ = ["History"]
