from redis import Redis
from .base import Sampler
import numpy as np
import random
import pickle
import cloudpickle
import logging
from time import sleep
logging.basicConfig(level="INFO")

worker_logger = logging.getLogger("REDIS-WORKER")

QUEUE = "queue"
N_EVAL = "n_eval"
N_PARTICLES = "n_particles"
SSA = "sample_simulate_accept"
N_WORKER = "n_workers"

MSG = "msg_pubsub"
START = "start"
SLEEP_TIME = .1


def work_on_population(redis: Redis):
    n_worker = redis.incr(N_WORKER)
    worker_logger.info("Start work population. I am worker {}"
                       .format(n_worker))
    sample, simulate, accept = pickle.loads(redis.get(SSA))

    random.seed()
    np.random.seed()
    n_particles = int(redis.get(N_PARTICLES).decode())

    internal_counter = 0
    while n_particles > 0:
        particle_id = redis.incr(N_EVAL)
        internal_counter += 1

        new_param = sample()
        new_sim = simulate(new_param)
        if accept(new_sim):
            n_particles = redis.decr(N_PARTICLES)
            redis.rpush(QUEUE, cloudpickle.dumps((particle_id, new_sim)))

    redis.decr(N_WORKER)
    worker_logger.info("Finished population, did {} samples."
                       .format(internal_counter))


def work(redis: Redis=None):
    worker_logger.info("Start redis worker")
    if redis is None:
        redis = Redis()

    p = redis.pubsub()
    p.subscribe(**{MSG: lambda x: work_on_population(redis)})

    for _ in p.listen():
        pass


class RedisEvalParallelSampler(Sampler):
    def __init__(self, redis: Redis=None):
        super().__init__()
        self.redis = redis if redis is not None else Redis()

    def sample_until_n_accepted(self, sample_one, simulate_one, accept_one, n):
        self.redis.set(SSA,
                       cloudpickle.dumps(
                           (sample_one, simulate_one, accept_one)))
        self.redis.set(N_EVAL, 0)
        self.redis.set(N_PARTICLES, n)
        self.redis.set(N_WORKER, 0)
        self.redis.delete(QUEUE)

        id_results = []

        self.redis.publish(MSG, START)

        while len(id_results) < n:
            dump = self.redis.blpop(QUEUE)[1]
            particle_with_id = pickle.loads(dump)
            id_results.append(particle_with_id)

        while int(self.redis.get(N_WORKER).decode()) > 0:
            sleep(SLEEP_TIME)

        # make sure all results are collected
        while self.redis.llen(QUEUE) > 0:
            id_results.append(pickle.loads(self.redis.blpop(QUEUE)[1]))

        # avoid bias toward short running evaluations
        id_results.sort(key=lambda x: x[0])
        id_results = id_results[:n]

        self.nr_evaluations_ = int(self.redis.get(N_EVAL).decode())

        population = [res[1] for res in id_results]
        return population


if __name__ == "__main__":
    work()
