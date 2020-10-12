import sys
import socket
import signal
from redis import StrictRedis
import pickle
import os
import cloudpickle
from time import time
import click
from .redis_logging import logger
from .cmd import (N_EVAL, N_ACC, N_REQ, ALL_ACCEPTED,
                  N_WORKER, SSA, QUEUE, START, STOP,
                  MSG, BATCH_SIZE, IS_PREL, GENERATION, idfy)
from multiprocessing import Pool
import numpy as np
import random


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


def work_on_population(redis: StrictRedis,
                       start_time: float,
                       max_runtime_s: float,
                       kill_handler: KillHandler):
    """
    Here the actual sampling happens.
    """
    # set timers
    population_start_time = time()
    cumulative_simulation_time = 0

    # read from pipeline
    pipeline = redis.pipeline()

    # get time
    t_b = redis.get(GENERATION)

    # check if anything was started already
    if t_b is None:
        return

    t = int(t_b.decode())

    # extract bytes
    ssa_b, batch_size_b, all_accepted_b, n_req_b, n_acc_b, is_prel_b \
        = (pipeline.get(idfy(SSA, t))
           .get(idfy(BATCH_SIZE, t))
           .get(idfy(ALL_ACCEPTED, t))
           .get(idfy(N_REQ, t))
           .get(idfy(N_ACC, t))
           .get(idfy(IS_PREL, t)).execute())

    if ssa_b is None:
        return

    kill_handler.exit = False

    if n_acc_b is None:
        return

    # convert from bytes
    simulate_one, sample_factory = pickle.loads(ssa_b)
    batch_size = int(batch_size_b.decode())
    all_accepted = bool(int(all_accepted_b.decode()))
    n_req = int(n_req_b.decode())
    is_prel = bool(int(is_prel_b.decode()))

    # notify sign up as worker
    n_worker = redis.incr(idfy(N_WORKER, t))
    logger.info(
        f"Begin generation {t}, batch size {batch_size}. "
        f"I am worker {n_worker}")

    # counter for number of simulations
    internal_counter = 0

    # create empty sample
    sample = sample_factory()

    # loop until no more particles required
    while int(redis.get(idfy(N_ACC, t)).decode()) < int(redis.get(idfy(N_REQ, t)).decode()) \
            and (not all_accepted or
                 int(redis.get(idfy(N_EVAL, t)).decode()) < n_req):
        if kill_handler.killed:
            logger.info(
                f"Worker {n_worker} received stop signal. "
                f"Terminating in the middle of a population "
                f"after {internal_counter} samples.")
            # notify quit
            redis.decr(idfy(N_WORKER, t))
            sys.exit(0)

        # check whether time's up
        current_runtime = time() - start_time
        if current_runtime > max_runtime_s:
            logger.info(
                f"Worker {n_worker} stops during population because "
                f"runtime {current_runtime} exceeds "
                f"max runtime {max_runtime_s}")
            # notify quit
            redis.decr(idfy(N_WORKER, t))
            return

        # check if is preliminary
        if is_prel and not bool(int(redis.get(idfy(IS_PREL, t)).decode())):
            # reload SSA object
            ssa_b = redis.get(idfy(SSA, t))
            simulate_one, sample_factory = pickle.loads(ssa_b)
            # cache
            is_prel = False

        # increase global number of evaluations counter
        particle_max_id = redis.incr(idfy(N_EVAL, t), batch_size)

        # timer for current simulation until batch_size acceptances
        this_sim_start = time()
        # collect accepted particles
        accepted_samples = []

        # make batch_size attempts
        for n_batched in range(batch_size):
            # increase evaluation counter
            internal_counter += 1
            try:
                # simulate
                new_sim = simulate_one()
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
                    # initialize new sample
                    sample = sample_factory()
            except Exception as e:
                logger.warning(f"Redis worker number {n_worker} failed. "
                               f"Error message is: {e}")
                # initialize new sample to be sure
                sample = sample_factory()

        # update total simulation-specific time
        cumulative_simulation_time += time() - this_sim_start

        # push to pipeline if at least one sample got accepted
        if len(accepted_samples) > 0:
            # new pipeline
            pipeline = redis.pipeline()
            # update particles counter if not preliminary
            if not is_prel:
                pipeline.incr(idfy(N_ACC, t), len(accepted_samples))
            # note: samples are appended 1-by-1
            pipeline.rpush(idfy(QUEUE, t), *accepted_samples)
            # execute all commands
            pipeline.execute()

    # end of sampling loop

    # notify quit
    redis.decr(idfy(N_WORKER, t))
    kill_handler.exit = True
    population_total_time = time() - population_start_time
    logger.info(
        f"Finished generation {t}, did {internal_counter} samples. "
        f"Simulation time: {cumulative_simulation_time:.2f}s, "
        f"total time {population_total_time:.2f}.")


@click.command(help="Evaluation parallel redis sampler for pyABC.")
@click.option('--host', default="localhost", help='Redis host.')
@click.option('--port', default=6379, type=int, help='Redis port.')
@click.option('--runtime', type=str, default="2h",
              help='Max worker runtime if the form <NR><UNIT>, '
                   'where <NR> is any number and <UNIT> can be s, '
                   '(S,) m, (M,) '
                   'h, (H,) d, (D) for seconds, minutes, hours and days. '
                   'E.g. for 12 hours you would pass --runtime=12h, for half '
                   'a day you could do 0.5d.')
@click.option('--password', default=None, help='Password for a secure '
                                               'connection.')
@click.option('--processes', type=int, default=1, help="The number of worker "
                                                       "processes to start")
def work(host="localhost",
         port=6379, runtime="2h",
         password=None,
         processes=1):
    """
    Corresponds to the entry point abc-redis-worker.
    """
    if processes == 1:
        # start a single process right here, not within pool
        # this handles the problem of starting a daemon process within a
        # daemon process
        return _work(host, port, runtime, password)

    with Pool(processes) as pool:
        res = pool.starmap(_work, [(host, port, runtime, password)]
                           * processes)
    return res


def _work(host="localhost", port=6379, runtime="2h", password=None):
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

        # check if it is int to (first iteration) run at least once
        if data == START or isinstance(data, int):
            work_on_population(redis, start_time, max_runtime_s, kill_handler)

        if data == STOP:
            logger.info("Received stop signal. Shutdown redis worker.")
            return

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
@click.option('--time', '-t', default=None, type=int, help="Generation t.")
@click.argument('command', type=str)
def manage(command, host="localhost", port=6379, password=None, time=None):
    """
    Corresponds to the entry point abc-redis-manager.
    """
    return _manage(command, host=host, port=port, password=password, time=time)


def _manage(command, host="localhost", port=6379, password=None, time=None):
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

    # default time is latest
    if time is None:
        time = int(redis.get(GENERATION).decode())

    if command == "info":
        pipeline = redis.pipeline()
        res = (pipeline.get(idfy(N_WORKER, time))
               .get(idfy(N_EVAL, time))
               .get(idfy(N_ACC, time))
               .get(idfy(N_REQ, time)).execute())
        res = [r.decode() if r is not None else r for r in res]
        print("Workers={} Evaluations={} Acceptances={}/{} (generation {})"
              .format(*res, time))
    elif command == "reset-workers":
        redis.set(idfy(N_WORKER, time), 0)


def is_server_used(redis: StrictRedis):
    """Check whether the server is currently in use."""
    # it would be cleaner to have a single flag that is set before a run and
    # unset after a run, but this solution should also work
    t_b = redis.get(GENERATION)
    if t_b is None:
        return False

    # the generation variable is not unset after a run
    t = int(t_b.decode())
    if not redis.exists(idfy(N_WORKER, t)):
        return False

    return True
