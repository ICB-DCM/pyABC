Model Selection
===============


Quickstart
----------

Here is a small example on how to do Bayesian model selection.



.. literalinclude:: ../examples/quickstart.py


You should see something similar to


.. literalinclude:: ../examples/quickstart.py.out

and maybe additional logging output such as


.. literalinclude:: ../examples/quickstart.py.err


Step by step explanation
------------------------

Defining a model
~~~~~~~~~~~~~~~~

To do model selection, we need first some models. A model is just a callable which takes a single dict as input
and returns a single dict as output. The keys of the input dictionary are the parameters of the model, the output
keys denote the summary statistics.
In the above example, the model is defined in line 7. The ``dict`` is passed as ``args`` and has the parameter ``x``
which denote in the above example the mean of the Gaussian. It returns the observed summary statistics ``y``, which
is just the sampled value.

For model selection we usually have more than one model. These are assembled in a list, as in line 12. We
require a Bayesian prior over the models, as defined in line 15 and a prior over the models' parameters as defined
in line 19. This concludes the model definition.

Configuring the ABCSMC
~~~~~~~~~~~~~~~~~~~~~~

Having the models defined, we can plug togetehr the ``ABCSMC`` class. Additionally, we need a model perturbation kernel
as defined in line 34 which governs with which probability to jump from one model class to another. Moreover,
we need a distance function (line 37), to measure the distance of obtained samples and a strategy for adjusting the
epsilon threshold (line 38).


Setting the observed data
~~~~~~~~~~~~~~~~~~~~~~~~~

Actually measured data can no be passed to the ABCSMC. In line 46, we have the actually observed (measured) data.
This is set in lie 47 via the ``set_data`` method. Additional meta information such as model names is also passed.
Moreover we have to set the output database.

Running the ABC
~~~~~~~~~~~~~~~

In line 53 we run the ``ABCSMC`` specifying the maximum number of populations and the epsilon value at which to terminate.
Whatever is reached first terminates the ABC run. The result is a :class:`History <abcsmc.History>` object which
can, for example be queried for the posterior probabilities.

