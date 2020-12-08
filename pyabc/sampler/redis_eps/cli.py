import sys
import socket
import signal
from redis import StrictRedis
import pickle
import os
import cloudpickle
from time import time
import click
from multiprocessing import Process
import numpy as np
import random
import logging

from .cmd import (N_EVAL, N_ACC, N_REQ, N_FAIL, ALL_ACCEPTED,
                  N_WORKER, SSA, QUEUE, START, STOP,
                  MSG, BATCH_SIZE, IS_LOOK_AHEAD, ANALYSIS_ID, GENERATION,
                  idfy)
from ..util import any_particle_preliminary

logger = logging.getLogger("Redis-Worker")

TIMES = {"s": 1,
         "m": 60,
         "h": 3600,
         "d":  24*3600}


def runtime_parse(s):
    unit = TIMES[s[-1].lower()]
    nr = float(s[:-1])
    return unit * nr


class KillHandler:
    def __init__(self):
        self.killed = False
        self.exit = True
        signal.signal(signal.SIGTERM, self.handle)
        signal.signal(signal.SIGINT, self.handle)

    def handle(self, *args):
        self.killed = True
        if self.exit:
            sys.exit(0)


def work_on_population(analysis_id: str,
                       t: int,
                       redis: StrictRedis,
                       catch: bool,
                       start_time: float,
                       max_runtime_s: float,
                       kill_handler: KillHandler):
    """
    Here the actual sampling happens.
    """
    def get_int(var: str):
        """Convenience function to read an int variable."""
        return int(redis.get(idfy(var, ana_id, t)).decode())

    # set timers
    population_start_time = time()
    cumulative_simulation_time = 0

    # read from pipeline
    pipeline = redis.pipeline()

    # short-form
    ana_id = analysis_id

    # extract bytes
    ssa_b, batch_size_b, all_accepted_b, is_look_ahead_b \
        = (pipeline.get(idfy(SSA, ana_id, t))
           .get(idfy(BATCH_SIZE, ana_id, t))
           .get(idfy(ALL_ACCEPTED, ana_id, t))
           .get(idfy(IS_LOOK_AHEAD, ana_id, t)).execute())

    # if the ssa object does not exist, something went wrong, return
    if ssa_b is None:
        return

    # only allow stopping the worker at particular points
    kill_handler.exit = False

    # convert from bytes
    simulate_one, sample_factory = pickle.loads(ssa_b)
    batch_size = int(batch_size_b.decode())
    all_accepted = bool(int(all_accepted_b.decode()))
    is_look_ahead = bool(int(is_look_ahead_b.decode()))

    # notify sign up as worker
    n_worker = redis.incr(idfy(N_WORKER, ana_id, t))
    logger.info(
        f"Begin generation {t}, batch size {batch_size}. "
        f"I am worker {n_worker}")

    # counter for number of simulations
    internal_counter = 0

    # create empty sample
    sample = sample_factory(is_look_ahead=is_look_ahead)

    # loop until no more particles required
    # all numbers are re-loaded in each iteration as they can dynamically
    #  update
    while get_int(N_ACC) < get_int(N_REQ) and (
            not all_accepted or
            get_int(N_EVAL) - get_int(N_FAIL) < get_int(N_REQ)):
        # check whether the process was externally asked to stop
        if kill_handler.killed:
            logger.info(
                f"Worker {n_worker} received stop signal. "
                "Terminating in the middle of a population "
                f"after {internal_counter} samples.")
            # notify quit
            redis.decr(idfy(N_WORKER, ana_id, t))
            sys.exit(0)

        # check whether time's up
        current_runtime = time() - start_time
        if current_runtime > max_runtime_s:
            logger.info(
                f"Worker {n_worker} stops during population because "
                f"runtime {current_runtime} exceeds "
                f"max runtime {max_runtime_s}")
            # notify quit
            redis.decr(idfy(N_WORKER, ana_id, t))
            # return to task queue
            return

        # check whether the analysis was terminated or replaced by a new one
        ana_id_new_b = redis.get(ANALYSIS_ID)
        if ana_id_new_b is None or str(ana_id_new_b.decode()) != ana_id:
            logger.info(
                f"Worker {n_worker} stops during population because "
                "the analysis seems to have been stopped.")
            # notify quit
            redis.decr(idfy(N_WORKER, ana_id, t))
            # return to task queue
            return

        # check if the analysis left the look-ahead mode
        if is_look_ahead and not bool(int(
                redis.get(idfy(IS_LOOK_AHEAD, ana_id, t)).decode())):
            # reload SSA object
            ssa_b = redis.get(idfy(SSA, ana_id, t))
            simulate_one, sample_factory = pickle.loads(ssa_b)
            # cache
            is_look_ahead = False
            # create new empty sample for clean split
            sample = sample_factory(is_look_ahead=is_look_ahead)

        # increase global evaluation counter (before simulation!)
        particle_max_id = redis.incr(idfy(N_EVAL, ana_id, t), batch_size)

        # timer for current simulation until batch_size acceptances
        this_sim_start = time()
        # collect accepted particles
        accepted_samples = []
        # whether any particle in this iteration is preliminary
        any_prel = False

        # make batch_size attempts
        for n_batched in range(batch_size):
            # increase evaluation counter
            internal_counter += 1
            try:
                # simulate
                new_sim = simulate_one()
            except Exception as e:
                logger.warning(f"Redis worker number {n_worker} failed. "
                               f"Error message is: {e}")
                # increment the failure counter
                redis.incr(idfy(N_FAIL, ana_id, t), 1)
                if not catch:
                    raise e
                continue

            # append to current sample
            sample.append(new_sim)
            # check for acceptance
            if new_sim.accepted:
                # the order of the IDs is reversed, but this does not
                # matter. Important is only that the IDs are specified
                # before the simulation starts

                # append to accepted list
                accepted_samples.append(
                    cloudpickle.dumps(
                        (particle_max_id - n_batched, sample)))
                any_prel = any_prel or any_particle_preliminary(sample)
                # initialize new sample
                sample = sample_factory(is_look_ahead=is_look_ahead)

        # update total simulation-specific time
        cumulative_simulation_time += time() - this_sim_start

        # push to pipeline if at least one sample got accepted
        if len(accepted_samples) > 0:
            # new pipeline
            pipeline = redis.pipeline()
            # update particles counter if nothing is preliminary,
            #  otherwise final acceptance is done by the sampler
            if not any_prel:
                pipeline.incr(idfy(N_ACC, ana_id, t), len(accepted_samples))
            # note: samples are appended 1-by-1
            pipeline.rpush(idfy(QUEUE, ana_id, t), *accepted_samples)
            # execute all commands
            pipeline.execute()

    # end of sampling loop

    # notify quit
    redis.decr(idfy(N_WORKER, ana_id, t))
    kill_handler.exit = True
    population_total_time = time() - population_start_time
    logger.info(
        f"Finished generation {t}, did {internal_counter} samples. "
        f"Simulation time: {cumulative_simulation_time:.2f}s, "
        f"total time {population_total_time:.2f}.")


@click.command(help="Evaluation parallel redis sampler for pyABC.")
@click.option('--host', default="localhost", help='Redis host.')
@click.option('--port', default=6379, type=int, help='Redis port.')
@click.option('--runtime', default="2h", type=str,
              help='Max worker runtime if the form <NR><UNIT>, '
                   'where <NR> is any number and <UNIT> can be s, '
                   '(S,) m, (M,) '
                   'h, (H,) d, (D) for seconds, minutes, hours and days. '
                   'E.g. for 12 hours you would pass --runtime=12h, for half '
                   'a day you could do 0.5d.')
@click.option('--password', default=None, type=str,
              help='Password for a secure connection.')
@click.option('--processes', default=1, type=int,
              help="The number of worker processes to start")
@click.option('--daemon', default=True, type=bool,
              help="Create subprocesses in daemon mode.")
@click.option('--catch', default=True, type=bool, help="Catch errors.")
def work(host="localhost",
         port=6379, runtime="2h",
         password=None,
         processes=1,
         daemon=True,
         catch=True):
    """Start workers.
    Corresponds to the entry point abc-redis-worker.
    """
    if processes == 1:
        # for a single process, no need to use any pooling
        return _work(host, port, runtime, password, daemon, catch)

    # define parallel processes
    procs = [
        Process(target=_work,
                args=(host, port, runtime, password, catch),
                daemon=daemon)
        for _ in range(processes)]

    # start them
    for proc in procs:
        proc.start()

    # log
    for proc in procs:
        logger.info(f"Started subprocess with pid {proc.pid}")

    # wait for them to return
    for proc in procs:
        proc.join()


def _work(host="localhost", port=6379, runtime="2h", password=None,
          catch=True):
    np.random.seed()
    random.seed()

    kill_handler = KillHandler()

    start_time = time()
    max_runtime_s = runtime_parse(runtime)
    logger.info(
        f"Start redis worker. Max run time {max_runtime_s}s, "
        f"HOST={socket.gethostname()}, PID={os.getpid()}")

    # connect to the redis server
    redis = StrictRedis(host=host, port=port, password=password)

    # subscribe to the server's MSG channel
    p = redis.pubsub()
    p.subscribe(MSG)

    # Block-wait for publications. Every message on the channel that has not
    #  been overridden yet is processed by all workers exactly once via
    #  listen(), even if the workers were started after publication.
    #  When the server is stopped, an error makes the workers stop too.
    for msg in p.listen():
        try:
            data = msg["data"].decode()
        except AttributeError:
            data = msg["data"]

        if data == START:
            # extract population definition
            #  analysis id
            analysis_id = str(redis.get(ANALYSIS_ID).decode())
            #  current time index
            t = int(redis.get(idfy(GENERATION, analysis_id)).decode())
            # work on the specified population
            work_on_population(
                analysis_id=analysis_id, t=t,
                redis=redis, catch=catch, start_time=start_time,
                max_runtime_s=max_runtime_s, kill_handler=kill_handler)

        elif data == STOP:
            logger.info("Received stop signal. Shutdown redis worker.")
            return

        # TODO other messages (some integers?) are ignored

        # check total time condition
        elapsed_time = time() - start_time
        if elapsed_time > max_runtime_s:
            logger.info(
                "Shutdown redis worker. Max runtime {}s reached"
                .format(max_runtime_s))
            return


@click.command(help="ABC Redis cluster manager. "
                    "The command can be 'info' or 'stop'. "
                    "For 'stop' the workers are shut down cleanly "
                    "after the current population. "
                    "For 'info' you'll see how many workers are connected, "
                    "how many evaluations the current population has, and "
                    "how many particles are still missing. "
                    "For 'reset-workers', the worker count will be resetted to"
                    "zero. This does not cancel the sampling. This is useful "
                    "if workers were unexpectedly killed.")
@click.option('--host', default="localhost", help="Redis host.")
@click.option('--port', default=6379, type=int, help="Redis port.")
@click.option('--password', default=None, type=str, help="Redis password.")
@click.option('--time', '-t', 't', default=None, type=int,
              help="Generation t.")
@click.argument('command', type=str)
def manage(command, host="localhost", port=6379, password=None, t=None):
    """Manage workers.
    Corresponds to the entry point abc-redis-manager.
    """
    return _manage(command, host=host, port=port, password=password, t=t)


def _manage(command, host="localhost", port=6379, password=None, t=None):
    if command not in ["info", "stop", "reset-workers"]:
        print("Unknown command: ", command)
        return

    redis = StrictRedis(host=host, port=port, password=password)
    if command == "stop":
        redis.publish(MSG, STOP)
        return

    # check whether an analysis is running
    if not is_server_used(redis):
        print("No active generation")
        return

    # id of the current analysis
    ana_id = str(redis.get(ANALYSIS_ID).decode())

    # default time is latest
    if t is None:
        t = int(redis.get(idfy(GENERATION, ana_id)).decode())

    if command == "info":
        pipeline = redis.pipeline()
        res = (pipeline.get(idfy(N_WORKER, ana_id, t))
               .get(idfy(N_EVAL, ana_id, t))
               .get(idfy(N_ACC, ana_id, t))
               .get(idfy(N_REQ, ana_id, t)).execute())
        res = [r.decode() if r is not None else r for r in res]
        print("Workers={} Evaluations={} Acceptances={}/{} (generation {})"
              .format(*res, t))
    elif command == "reset-workers":
        redis.set(idfy(N_WORKER, ana_id, t), 0)


def is_server_used(redis: StrictRedis):
    """Check whether the server is currently in use."""
    analysis_id = redis.get(ANALYSIS_ID)
    if analysis_id is None:
        return False
    return True
