from redis import StrictRedis
from .base import Sampler
import pickle
import cloudpickle
import logging
from time import sleep, time
import click
import signal
import sys
logging.basicConfig(level="INFO")

worker_logger = logging.getLogger("REDIS-WORKER")

QUEUE = "queue"
N_EVAL = "n_eval"
N_PARTICLES = "n_particles"
SSA = "sample_simulate_accept"
N_WORKER = "n_workers"

MSG = "msg_pubsub"
START = "start"
STOP = "stop"
SLEEP_TIME = .1


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


def work_on_population(redis: StrictRedis, kill_handler: KillHandler):
    ssa = redis.get(SSA)
    if ssa is None:
        return
    kill_handler.exit = False
    n_worker = redis.incr(N_WORKER)
    worker_logger.info("Begin population. I am worker {}"
                       .format(n_worker))
    sample, simulate, accept = pickle.loads(ssa)

    n_particles = int(redis.get(N_PARTICLES).decode())

    internal_counter = 0
    while n_particles > 0:
        if kill_handler.killed:
            worker_logger.info("Worker {} received stop signal. "
                               "Terminating in the middle of a population"
                               " after {} samples."
                               .format(n_worker, internal_counter))
            redis.decr(N_WORKER)
            sys.exit(0)

        particle_id = redis.incr(N_EVAL)
        internal_counter += 1

        new_param = sample()
        new_sim = simulate(new_param)
        if accept(new_sim):
            n_particles = redis.decr(N_PARTICLES)
            redis.rpush(QUEUE, cloudpickle.dumps((particle_id, new_sim)))
        else:
            n_particles = int(redis.get(N_PARTICLES).decode())

    redis.decr(N_WORKER)
    kill_handler.exit = True
    worker_logger.info("Finished population, did {} samples."
                       .format(internal_counter))


@click.command(help="Evaluation parallel redis sampler for pyABC.")
@click.option('--host', default="localhost", help='Redis host.')
@click.option('--port', default=6379, type=int, help='Redis port.')
@click.option('--runtime', type=str, default="2h",
              help='Max worker runtime in seconds.')
def work(host="localhost", port=6379, runtime="2h"):
    kill_handler = KillHandler()

    start_time = time()
    max_runtime_s = runtime_parse(runtime)
    worker_logger.info("Start redis worker. Max run time {}s"
                       .format(max_runtime_s))
    redis = StrictRedis(host=host, port=port)

    p = redis.pubsub()
    p.subscribe(MSG)

    work_on_population(redis, kill_handler)
    listener = p.listen()
    next(listener)  # first message contains as data only 1 number 1
    for msg in listener:
        if msg["data"].decode() == START:
            work_on_population(redis, kill_handler)
        elapsed_time = time() - start_time
        if msg["data"].decode() == STOP:
            worker_logger.info("Received stop signal. Shutdown redis worker.")
            return
        if elapsed_time > max_runtime_s:
            worker_logger.info("Shutdown redis worker. Max runtime {}s reached"
                               .format(max_runtime_s))
            return


@click.command(help="ABC Redis cluster manager. "
                    "The command can be 'info' or 'stop'. "
                    "For 'stop' the workers are shut down cleanly "
                    "after the current population. "
                    "For 'info' you'll see how many workers are connected, "
                    "how many evaluations the current population has, and "
                    "how many particles are still missing.")
@click.option('--host', default="localhost", help='Redis host.')
@click.option('--port', default=6379, type=int, help='Redis port.')
@click.argument('command', type=str)
def manage(command, host="localhost", port=6379):
    redis = StrictRedis(host=host, port=port)
    if command == "info":
        pipe = redis.pipeline()
        pipe.get(N_WORKER)
        pipe.get(N_EVAL)
        pipe.get(N_PARTICLES)
        res = pipe.execute()
        res = [r.decode() if r is not None else r for r in res]
        print("Workers={} Evaluations={} Particles={}".format(*res))
    elif command == "stop":
        redis.publish(MSG, STOP)
    else:
        print("Unknown command:", command)


class RedisEvalParallelSampler(Sampler):
    """
    Redis based low latency sampler.
    This sampler is extremely well performiing in distributed environments.
    It vastly outperforms :class:`pyabc.parallel.sampler.DaskDistributedSampler` for
    short model evaluation runtimes. The longer the model evaluation times,
    the less the advantage becomes. It requires a running redis server as
    broker.

    This sampler requires workers to be started via the command
    ``abc-redis-worker``.
    An example call might look like
    ``abc-redis-worker --host=123.456.789.123 --max_runtime_s=7200``
    to connect to a Redis server on IP ``123.456.789.123`` and to terminate
    the worker after finishing the first population which ends after 7200s
    since worker start. So the actual runtime might be longer thatn 7200s.
    See ``abc-redis-worker --help`` for its options.

    Use the command ``abc-redis-manager`` to retrieve info and stop the running
    workers.

    Parameters
    ----------

    host: str, optional
        IP address or name of the Redis server.
        Default is "localhost"

    port: int, optional
        Port of the Redis server.
        Default if 6379.
    """
    def __init__(self, host="localhost", port=6379):
        super().__init__()
        worker_logger.info("Redis sampler: host={} port={}".format(host, port))
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
    manage()
