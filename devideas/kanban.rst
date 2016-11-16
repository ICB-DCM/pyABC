Kanban board
============

https://en.wikipedia.org/wiki/Kanban_board

Items are sorted in decreasing order of importance. The top most is the most important one.



Backlog
-------

- Benchmarks for adaptive population sizes. Show that this helps.
- Refactor ``pyabc.storage.History`` and ``pyabc.loader.ABCLoader``
    - Both deal with databas access. They sould be a single clas
    - Remove connectomics specific stuff from ``ABCLoader``
    - Remove things like covariance calculation from ``History``
    - Make sure that ``History.append_population`` stores to database.
      The current double implementation of local cache and database is weird.
- Good performance for heterogeneous evaluation times
- Good performance for small evaluation times
- Good performance for large evaluation times
- Hardware failure



In Progress
-----------

- **DR** Static worker pools?
    - so... let's try centralized generation by a master and distribution by celery.
      seems to strike the best balance between simplicity and re-using established tools.
    - target up to 100 jobs per second
    - Assume 10s per simulation on a single core. assume cluster of 1000 cores.
      So about 100 samples per second overall.
- **EK** adaptive population size
    - ✓ max nr particles limit
    - ✓ deal with corner cases of no particles, 1 particle, 2 particles
    - ✓ deal with particles with no parameters (for model selection with models without paramters)
- **EK**n DB refactor and rethinking
    - design choices
        - stick with relational database.
            - relational databases are better suited when we don't know the queries beforehand
            - nosql databases need/should to be designed with the queries already in mind
    - use sqlalchemy to support many different database types, e.g. sqlite, mysql, postgres, etc
    - also this should be the least amount of work as current db backend is relational

Review
------

- Model interface


Done
----

- **DR** Create Sampler interface
- **EK** Create particle perturbation interface