Thoughts
========

* vorschlag: 1:1 partikel evaluation und qsub job
  * einfach zu implmentieren
  * netzwerkausfall robust
  * overhead fuer kurze simulationen
* vorschlag: worker
  * mehr verwaltungsaufwand
  * weninger overhead fuer kurze simulationen
  * ja nach implementierung robust oder nicht gegen workerausfall
    * celery mit rabbitmq ist robust
    * ipcluster nicht robust
    * problem: keine freien slots fuer worker oder aehnliches
* job cancel?
  * do we need to kill a job
  * job runtimes until half a day
    * we might need to kill the jobs
* job cancel optional. mode with cancel and mode without cancel
* job ausfuehrung in eigener python funktion
  * need to kill the worker
* https://wiki.cebitec.uni-bielefeld.de/bibiserv-1.25.2/index.php/BiBiGrid
* kill all workers with qdel?
* don't forget acceptance rate

Minimal requirements
====================

*