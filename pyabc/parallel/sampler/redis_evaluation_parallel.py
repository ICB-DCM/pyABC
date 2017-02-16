from redis import StrictRedis
from .base import Sampler
import numpy as np
import random
import pickle
import cloudpickle
import logging
from time import sleep, time
import click
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


def work_on_population(redis: StrictRedis):
    ssa = redis.get(SSA)
    if ssa is None:
        return
    n_worker = redis.incr(N_WORKER)
    worker_logger.info("Begin population. I am worker {}"
                       .format(n_worker))
    sample, simulate, accept = pickle.loads(ssa)

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


@click.command(help="Evaluation parallel redis sampler for pyABC.")
@click.option('--host', default="localhost", help='Redis host.')
@click.option('--port', default=6379, type=int, help='Redis port.')
@click.option('--max_runtime_s', type=int, default=50*3600,
              help='Max worker runtime in seconds.')
def work(host="localhost", port=6379, max_runtime_s=50*3600):
    start_time = time()
    worker_logger.info("Start redis worker. Max run time {}s"
                       .format(max_runtime_s))
    redis = StrictRedis(host=host, port=port)


    p = redis.pubsub()
    p.subscribe(MSG)

    work_on_population(redis)
    listener = p.listen()
    next(listener)  # first message contains as data only 1 number 1
    for msg in listener:
        if msg["data"].decode() == "start":
            work_on_population(redis)
        elapsed_time = time() - start_time
        if elapsed_time > max_runtime_s:
            worker_logger.info("Shutdown redis worker. Max runtime {}s reached"
                               .format(max_runtime_s))
            return


class RedisEvalParallelSampler(Sampler):
    def __init__(self, host="localhost", port=6379):
        super().__init__()
        self.redis = StrictRedis(host=host, port=port)

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

        self.nr_evaluations_ = int(self.redis.get(N_EVAL).decode())

        self.redis.delete(SSA)
        self.redis.delete(N_EVAL)
        self.redis.delete(N_PARTICLES)
        # avoid bias toward short running evaluations
        id_results.sort(key=lambda x: x[0])
        id_results = id_results[:n]



        population = [res[1] for res in id_results]
        return population


if __name__ == "__main__":
    work()
