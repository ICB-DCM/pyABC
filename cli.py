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
                  MSG, BATCH_SIZE, IS_PREL, GENERATION)
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
                       start_time: int,
                       max_runtime_s: int,
                       kill_handler: KillHandler):
    """
    Here the actual sampling happens.
    """


    # set timers
    population_start_time = time()
    cumulative_simulation_time = 0

    # read from pipeline
    pipeline = redis.pipeline()

    t = int(pipeline.get(GENERATION).decode())
    # extract bytes
    ssa_b, batch_size_b, all_accepted_b, n_req_b, n_acc_b, is_prel_b \
        = (pipeline.get(SSA + "_" + str(t)).get(BATCH_SIZE)
           .get(ALL_ACCEPTED + "_" + str(t)).get(N_REQ + "_" + str(t))
           .get(N_ACC + "_" + str(t)).get(IS_PREL+"_"+str(t)).execute())

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
    is_prel = bool(int(is_prel_b.decode))

    # notify sign up as worker
    n_worker = redis.incr(N_WORKER + "_" + str(t))
    logger.info(
        f"Begin population, batch size {batch_size}. "
        f"I am worker {n_worker}")

    # counter for number of simulations
    internal_counter = 0

    # create empty sample
    sample = sample_factory()

    # loop until no more particles required
    while int(redis.get(N_ACC + "_" + str(t)).decode()) < n_req \
            and (not all_accepted or int(redis.get(N_EVAL + "_" + str(t)).decode()) < n_req):
        if kill_handler.killed:
            logger.info(
                f"Worker {n_worker} received stop signal. "
                f"Terminating in the middle of a population "
                f"after {internal_counter} samples.")
            # notify quit
            redis.decr(N_WORKER + "_" + str(t))
            sys.exit(0)

        # check whether time's up
        current_runtime = time() - start_time
        if current_runtime > max_runtime_s:
            logger.info(
                f"Worker {n_worker} stops during population because "
                f"runtime {current_runtime} exceeds "
                f"max runtime {max_runtime_s}")
            # notify quit
            redis.decr(N_WORKER + "_" + str(t))
            return

        # check whether preliminary data is still active and update if not
        if is_prel is True:
            if bool(int(pipeline.get(IS_PREL + "_" + str(t)).decode())) is False:
                is_prel = False
                ssa_b = pipeline.get(SSA + "_" + str(t))
                simulate_one, sample_factory = pickle.loads(ssa_b)

        # increase global number of evaluations counter
        particle_max_id = redis.incr(N_EVAL + "_" + str(t), batch_size)

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
            # update particles counter
            pipeline.incr(N_ACC + "_" + str(t), len(accepted_samples))
            # note: samples are appended 1-by-1
            pipeline.rpush(QUEUE + "_" + str(t), *accepted_samples)
            # execute all commands

    # end of sampling loop

    # notify quit
    redis.decr(N_WORKER + "_" + str(t))
    kill_handler.exit = True
    population_total_time = time() - population_start_time
    logger.info(
        f"Finished population, did {internal_counter} samples. "
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
    redis = StrictRedis(host=host, port=port, password=password)

    p = redis.pubsub()
    p.subscribe(MSG)
    listener = p.listen()
    for msg in listener:
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
@click.option('--host', default="localhost", help='Redis host.')
@click.option('--port', default=6379, type=int, help='Redis port.')
@click.option('--password', default=None, type=str, help='Redis password.')
@click.argument('command', type=str)
def manage(command, host="localhost", port=6379, password=None):
    """
    Corresponds to the entry point abc-redis-manager.
    """
    return _manage(command, host=host, port=port, password=password)


def _manage(command, host="localhost", port=6379, password=None):
    redis = StrictRedis(host=host, port=port, password=password)
    pipe = redis.pipeline()
    t = pipe.get(GENERATION)
    if command == "info":
        pipe.get(N_WORKER+"_"+str(t))
        pipe.get(N_EVAL+"_"+str(t))
        pipe.get(N_ACC+"_"+str(t))
        pipe.get(N_REQ+"_"+str(t))
        res = pipe.execute()
        res = [r.decode() if r is not None else r for r in res]
        print("Workers={} Evaluations={} Acceptances={}/{}".format(*res))
    elif command == "stop":
        redis.publish(MSG, STOP)
    elif command == "reset-workers":
        redis.set(N_WORKER+"_"+str(t), 0)
    else:
        print("Unknown command:", command)
