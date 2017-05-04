What is pyABC about?
====================

pyABC helps to solve the problem of parameter inference, in settings where all
you can do is simulate from the (black-box) model but no further analysis is
possible. Putting it differently, if you have a forward simulator, then pyABC
does the backwards parameter inference step for you.


.. image:: abc_general.svg
   :alt: ABC
   :width: 100%



What you need
-------------

* Some kind of experimentally observed or synthetically generated data
* a parametrized stochastic simulator (e.g. a Python functions which makes
  calls to random nunmber generators)
  which supposedly explains the data


What you don't need
-------------------

* the likelihood function: p(parameter|data) is *not* required.




When better not to use pyABC
----------------------------

.. figure:: rose_hammer.svg
   :width: 50%
   :alt: ABC
   :align: left

   Not everything is a nail.

ABC likelihood inference is not a hammer for every nail. If you're actually able to write
down the likelihood then use it and apply a different inference technique which exploits it.
This package helps to solve the much harder problem of likelihood free inference.






Why to use pyABC?
-----------------



This is a package for Approximate Bayesian Computation, using a Sequential Monte Carlo scheme.
This provides a particularly efficient technique for Bayesian posterior estimation in cases where
it is very hard to calculate the likelihood function efficiently.

.. image:: multicore_distributed.svg
   :alt: Multicore and distributed
   :width: 100%


pyABC was designed to perform well on

* multicore and
* distributed environments.

pyABC integrates well with SGE like environments, as commonly found in scientific settings,
but can also be deployed to the cloud. Amongst other backend modes,
`Dask.distributed <http://distributed.readthedocs.io/en/latest/>`_  can be used under the hood.


It sounds like a contradiction, but pyABC is on the one hand easy to use for standard applications,
on the other hand it allows for flexible experimentation, exploring all aspects of new ABC-SMC schemes.
Apart of a rich set of default choices, it is easy to parametrize aspects of your algorithm through the implementation
of custom classes.