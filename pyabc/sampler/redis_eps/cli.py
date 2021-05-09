import socket
from redis import StrictRedis
import os
from time import time
import click
from multiprocessing import Process
import numpy as np
import random
import logging

from .cmd import (N_EVAL, N_ACC, N_REQ, N_WORKER, START, STOP, MSG,
                  ANALYSIS_ID, GENERATION, MODE, DYNAMIC, STATIC, idfy)
from .util import KillHandler
from .work import work_on_population_dynamic
from .work_static import work_on_population_static

logger = logging.getLogger("ABC.Sampler")

TIMES = {"s": 1,
         "m": 60,
         "h": 3600,
         "d":  24*3600}


def runtime_parse(s):
    unit = TIMES[s[-1].lower()]
    nr = float(s[:-1])
    return unit * nr


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
        return _work(host, port, runtime, password, catch)

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

        if data == START or isinstance(data, int):
            # Sometimes, redis weirdly only publishes an int (1) at the
            #  beginning, but not the actual START message if the workers
            #  were started later.
            #  Therefore make sure all variables are really there.

            # analysis id
            analysis_id_b = redis.get(ANALYSIS_ID)
            if analysis_id_b is None:
                continue
            analysis_id = str(analysis_id_b.decode())

            #  current time index
            t_b = redis.get(idfy(GENERATION, analysis_id))
            if t_b is None:
                continue
            t = int(t_b.decode())

            # parallelization mode
            mode_b = redis.get(idfy(MODE, analysis_id, t))
            if mode_b is None:
                continue
            mode = str(mode_b.decode())

            # work on the specified population in dynamic or static mode
            if mode == DYNAMIC:
                work_on_population_dynamic(
                    analysis_id=analysis_id, t=t,
                    redis=redis, catch=catch, start_time=start_time,
                    max_runtime_s=max_runtime_s, kill_handler=kill_handler)
            elif mode == STATIC:
                work_on_population_static(
                    analysis_id=analysis_id, t=t,
                    redis=redis, catch=catch, start_time=start_time,
                    max_runtime_s=max_runtime_s, kill_handler=kill_handler)
            else:
                # this should never happen
                raise ValueError(f"Did not recognize mode {mode}")

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
