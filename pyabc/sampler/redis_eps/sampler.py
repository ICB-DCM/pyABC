import numpy as np
import pickle
from time import sleep
import cloudpickle
from redis import StrictRedis
from typing import Callable
from jabbar import jabbar

from ...sampler import Sampler
from .cmd import (SSA, N_EVAL, N_ACC, N_REQ, ALL_ACCEPTED,
                  N_WORKER, QUEUE, MSG, START,
                  SLEEP_TIME, BATCH_SIZE, IS_PREL, GENERATION, idfy)
from .redis_logging import logger


class RedisEvalParallelSampler(Sampler):
    """
    Redis based low latency sampler.
    This sampler is well performing in distributed environments.
    It is usually faster than the
    :class:`pyabc.sampler.DaskDistributedSampler` for
    short model evaluation runtimes. The longer the model evaluation times,
    the less the advantage becomes. It requires a running Redis server as
    broker.

    This sampler requires workers to be started via the command
    ``abc-redis-worker``.
    An example call might look like
    ``abc-redis-worker --host=123.456.789.123 --runtime=2h``
    to connect to a Redis server on IP ``123.456.789.123`` and to terminate
    the worker after finishing the first population which ends after 2 hours
    since worker start. So the actual runtime might be longer than 2h.
    See ``abc-redis-worker --help`` for its options.

    Use the command ``abc-redis-manager`` to retrieve info and stop the running
    workers.

    Start as many workers as you wish. Workers can be dynamically added
    during the ABC run.

    Parameters
    ----------

    host: str, optional
        IP address or name of the Redis server.
        Default is "localhost".

    port: int, optional
        Port of the Redis server.
        Default is 6379.

    password: str, optional
        Password for a protected server. Default is None (no protection).

    batch_size: int, optional
        Number of model evaluations the workers perform before contacting
        the REDIS server. Defaults to 1. Increase this value if model
        evaluation times are short or the number of workers is large
        to reduce communication overhead.
    """
    def __init__(self,
                 host: str = "localhost",
                 port: int = 6379,
                 password: str = None,
                 batch_size: int = 1,
                 look_ahead: bool = False):
        super().__init__()
        logger.debug(
            f"Redis sampler: host={host} port={port}")
        # handles the connection to the redis-server
        self.redis: StrictRedis = StrictRedis(
            host=host, port=port, password=password)
        self.batch_size: int = batch_size
        self.look_ahead: bool = look_ahead

    def n_worker(self):
        """
        Get the number of connected workers.

        Returns
        -------

        Number of workers connected.
        """
        return self.redis.pubsub_numsub(MSG)[0][-1]

    def sample_until_n_accepted(
            self, n, simulate_one, t, analysis_info,
            max_eval=np.inf, all_accepted=False):
        if self.generation_t_was_started(t):
            # update the SSA function
            self.redis.set(
                idfy(SSA, t),
                cloudpickle.dumps((simulate_one, self.sample_factory)))
        else:
            # set up all variables for the new generation
            self.start_generation_t(
                n=n, t=t, simulate_one=simulate_one, all_accepted=all_accepted,
                is_prel=False)

        id_results = []

        # wait until n acceptances
        with jabbar(total=n, enable=self.show_progress, keep=False) as bar:
            while len(id_results) < n:
                # pop result from queue, block until one is available
                dump = self.redis.blpop(idfy(QUEUE, t))[1]
                # extract pickled object
                particle_with_id = pickle.loads(dump)
                # TODO check whether the acceptance criterion changed
                # append to collected results
                id_results.append(particle_with_id)
                bar.inc()

        # wait until all workers done
        while int(self.redis.get(idfy(N_WORKER, t)).decode()) > 0:
            sleep(SLEEP_TIME)

        # make sure all results are collected
        while self.redis.llen(idfy(QUEUE, t)) > 0:
            id_results.append(
                pickle.loads(self.redis.blpop(idfy(QUEUE, t))[1]))

        # set total number of evaluations
        self.nr_evaluations_ = int(self.redis.get(idfy(N_EVAL, t)).decode())

        # remove all time-specific variables
        self.clear_generation_t(t)

        # avoid bias toward short running evaluations (for
        # dynamic scheduling)
        id_results.sort(key=lambda x: x[0])
        id_results = id_results[:n]

        results = [res[1] for res in id_results]

        # create 1 to-be-returned sample from results
        sample = self._create_empty_sample()
        for j in range(n):
            sample += results[j]

        return sample

    def start_generation_t(
            self, n: int, t: int, simulate_one: Callable, all_accepted: bool,
            is_prel: bool):
        """Start generation `t`."""
        # write initial values to pipeline
        pipeline = self.redis.pipeline()

        # initialize variables for time t
        self.redis.set(idfy(SSA, t),
                       cloudpickle.dumps((simulate_one, self.sample_factory)))
        pipeline.set(idfy(N_EVAL, t), 0)
        pipeline.set(idfy(N_ACC, t), 0)
        pipeline.set(idfy(N_REQ, t), n)
        pipeline.set(idfy(ALL_ACCEPTED, t), int(all_accepted))  # encode as int
        pipeline.set(idfy(N_WORKER, t), 0)
        pipeline.set(idfy(BATCH_SIZE, t), self.batch_size)
        pipeline.set(idfy(IS_PREL, t), int(is_prel))  # encode as int

        # update the current generation variable
        pipeline.set(GENERATION, t)

        # execute all commands
        pipeline.execute()

        # publish start message
        self.redis.publish(MSG, START)

    def generation_t_was_started(self, t: int):
        """Check whether generation `t` was started already."""
        # just check any of the variables for time t
        return self.redis.exists(idfy(N_REQ, t))

    def clear_generation_t(self, t: int):
        """Clean up after generation `t` has finished."""
        # delete keys from pipeline
        pipeline = self.redis.pipeline()
        pipeline.delete(idfy(SSA, t))
        pipeline.delete(idfy(N_EVAL, t))
        pipeline.delete(idfy(N_ACC, t))
        pipeline.delete(idfy(N_REQ, t))
        pipeline.delete(idfy(ALL_ACCEPTED, t))
        pipeline.delete(idfy(N_WORKER, t))
        pipeline.delete(idfy(BATCH_SIZE, t))
        pipeline.delete(idfy(QUEUE, t))
        pipeline.execute()
