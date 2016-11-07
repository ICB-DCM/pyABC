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



Task queues evaluation
======================

- Celery task queue: Up to 500 task submissions per second were possible with redis broker.
    - easy to set up with redis broker
    - rabbitmq broker is not that much fun: you need an erlang virtual machine, as well as root access to run rabbitmq.
      (although there are ways around that. but it requires fiddling)
- pure redis solution
    - much faster, up to 5000 task submissions per second possible. **BUT** not with the same features.
      I.e. no hearbeaat, no message acknowledgments, not that level of worker failure safety
- conceptual I see with the "task queue perspective"
    - tasks are generated on some master
    - master has to generate a number of tasks depending on worker availability,
      i.e. we need a master which is aware of the numer of registered workders
- conceptually simpler solution (IMHO), "distributed generators perspective"
    - master only publishes function and requiered nr of accepted evaluations
        - function published as pickle byte string
        - function cached on workers
    - workers generate particles as long as necessary
    - this ensures automatic scaling with nr registered workers
    - master does not have to generate more proposals if more workers are registered
    - the workers themselves would also put the accepted particles
      in an accepted (ordered) set or list and discard invalid ones
        - minimal master interaction
    - however, some kind of heartbeat and worker restart is required.
    - workers could periodically check for failed workers and take over their jobs
        - again, this could work without master interaction
        - more decentralized organization, less master side orchestration required
    - this is semo master failure save: redis can do dumps, so we can resume, but we would loose something
        - this could be acceptable
- a viable way I see:
    - do a custom redis based solution
    - no "task queue perspective", but "distributed generators perspective"


