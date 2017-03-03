Quickstart
==========


Exemplary model and parameter selection
---------------------------------------

Here is a small example on how to do Bayesian model selection using
the following classes from the pyABC package:

* :class:`ABCSMC <pyabc.smc.ABCSMC>`
* :class:`RV <pyabc.random_variables.RV>`
* :class:`Distribution <pyabc.random_variables.Distribution>`
* :class:`PercentileDistanceFunction <pyabc.distance_functions.PercentileDistanceFunction>`
* :class:`ConstantPopulationStrategy <pyabc.populationstrategy.ConstantPopulationStrategy>`.


.. literalinclude:: ../examples/quickstart.py
   :linenos:


You should see something similar to


.. literalinclude:: ../examples/quickstart.py.out

and maybe additional logging output.


Step by step explanation
------------------------

Defining a model
~~~~~~~~~~~~~~~~

To do model selection, we need first some models. A model, in the simplest case,
is just a callable which takes a single ``dict`` as input
and returns a single ``dict`` as output. The keys of the input dictionary are the parameters of the model, the output
keys denote the summary statistics.
In the above example, the model is defined in line 13.
The ``dict`` is passed as ``args`` and has the parameter ``x``
which denote in the above example the mean of the Gaussian.
It returns the observed summary statistics ``y``, which
is just the sampled value.

For model selection we usually have more than one model.
These are assembled in a list, as in line 18 and 19. We
require a Bayesian prior over the models, as defined in line 23.
The default is to have a uniform prior over the model classes.
This concludes the model definition.

Configuring the ABCSMC
~~~~~~~~~~~~~~~~~~~~~~

Having the models defined, we can plug together the ``ABCSMC`` class.
We need a distance function (line 33),
to measure the distance of obtained samples.


Setting the observed data
~~~~~~~~~~~~~~~~~~~~~~~~~

Actually measured data can now be passed to the ABCSMC.
In line 39, we have the actually observed (measured) data.
This is set in line 40 via the ``set_data`` method.
Additional meta information such as model names is also passed.
Moreover we have to set the output database, as defined in line 37.


Running the ABC
~~~~~~~~~~~~~~~

In line 44 we run the ``ABCSMC`` specifying the epsilon value at which to terminate.
The default epsilon strategy is hte :class:`pyabc.epsilon.MedianEpsilon.`
Whatever is reached first, the epsilon or the maximum number allowed populations,
terminates the ABC run. The result is a :class:`History <pyabc.storage.History>` object which
can, for example be queried for the posterior probabilities.

