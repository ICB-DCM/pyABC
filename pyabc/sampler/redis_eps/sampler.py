import pickle
from time import sleep
import cloudpickle
from redis import StrictRedis
from ...sampler import Sampler
from .cmd import (SSA, N_EVAL, N_PARTICLES, N_WORKER, QUEUE, MSG, START,
                  SLEEP_TIME, BATCH_SIZE)
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

    batch_size: int, optional
        Number of model evaluations the workers perform before contacting
        the REDIS server. Defaults to 1. Increase this value if model
        evaluation times are short or the number of workers is large
        to reduce communication overhead.
    """
    def __init__(self, host="localhost", port=6379, batch_size=1):
        super().__init__()
        logger.debug(
            f"Redis sampler: host={host} port={port}")
        # handles the connection to the redis-server
        self.redis = StrictRedis(host=host, port=port)
        self.batch_size = batch_size

    def n_worker(self):
        """
        Get the number of connected workers.

        Returns
        -------

        Number of workers connected.
        """
        return self.redis.pubsub_numsub(MSG)[0][-1]

    def sample_until_n_accepted(self, n, simulate_one):
        # open pipeline
        pipeline = self.redis.pipeline()

        # write initial values to pipeline
        self.redis.set(
            SSA, cloudpickle.dumps((simulate_one, self.sample_factory)))
        pipeline.set(N_EVAL, 0)
        pipeline.set(N_PARTICLES, n)
        pipeline.set(N_WORKER, 0)
        pipeline.set(BATCH_SIZE, self.batch_size)
        # delete previous results
        pipeline.delete(QUEUE)
        # execute all commands
        pipeline.execute()

        id_results = []

        # publish start message
        self.redis.publish(MSG, START)

        # wait until n acceptances
        while len(id_results) < n:
            # pop result from queue, block until one is available
            dump = self.redis.blpop(QUEUE)[1]
            # extract pickled object
            particle_with_id = pickle.loads(dump)
            # append to collected results
            id_results.append(particle_with_id)

        # wait until all workers done
        while int(self.redis.get(N_WORKER).decode()) > 0:
            sleep(SLEEP_TIME)

        # make sure all results are collected
        while self.redis.llen(QUEUE) > 0:
            id_results.append(pickle.loads(self.redis.blpop(QUEUE)[1]))

        # set total number of evaluations
        self.nr_evaluations_ = int(self.redis.get(N_EVAL).decode())

        # delete keys from pipeline
        pipeline = self.redis.pipeline()
        pipeline.delete(SSA)
        pipeline.delete(N_EVAL)
        pipeline.delete(N_PARTICLES)
        pipeline.delete(BATCH_SIZE)
        pipeline.execute()

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
