Kanban board
============

https://en.wikipedia.org/wiki/Kanban_board

Items are sorted in decreasing order of importance. The top most is the most important one.



Backlog
-------

* Refactor ``pyabc.storage.History`` and ``pyabc.loader.ABCLoader``.
  * Both deal with databas access. They sould be a single clas
  * Remove connectomics specific stuff from ``ABCLoader``
  * Remove things like covariance calculation from ``History``
  * Make sure that ``History.append_population`` stores to database. The current
    double implementation of local cache and database is weird.
* Adaptive population size


In Progress
-----------

* **DR, EK** Static worker pools?


Review
------

* Model interface


Done
----

* **DR** Create Sampler interface
* **EK** Create particle perturbation interface