What is this package about?
===========================


When to use it?
---------------

This is a package for Approximate Bayesian Computationh, using a Sequential Monte Carlo scheme.
This provides a prticularly efficient technique for Bayesian posterior estimation in cases where
it is very hard to calculate the likelihood function efficient.



Why to use it?
--------------

This package was designed to perform well on

* single core single machine,
* multicore single machine and
* distributed environments.

It integrates well with SGE like environments, as commonly found in scientific settings,
but can also be deployed to cloud settings and can use
`Dask.distributed <http://distributed.readthedocs.io/en/latest/>`_ under the hood.


It sounds like a contradiction, but this package is on the one hand easy to use for standard applications,
on the other hand it can be very flexible experimented with it, exploring all aspects of new ABC-SMC schemes.
Appart of a rich set of default choices, it is easy to parametrize aspects of your algarithm through the implementation
of custom classes.