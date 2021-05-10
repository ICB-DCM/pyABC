from time import sleep
from subprocess import Popen  # noqa: S404
from multiprocessing import Process
import tempfile
import psutil
import time

from .cli import work, _manage
from .sampler import RedisEvalParallelSampler
from .sampler_static import RedisStaticSampler


class RedisServerStarter:

    def __init__(self,
                 password: str = None,
                 workers: int = 2,
                 processes_per_worker: int = 1,
                 daemon: bool = True,
                 catch: bool = True):
        # start server
        conn = psutil.net_connections()
        ports = [c.laddr[1] for c in conn]
        port = max(ports) + 1
        self.port = port
        self.password = password

        # create config file
        maybe_redis_conf = []
        if password is not None:
            fname = tempfile.mkstemp()[1]
            with open(fname, 'w') as f:
                f.write(f"requirepass {password}\n")
            maybe_redis_conf = [fname]

        self.redis_server = Popen(  # noqa: S607,S603
            ["redis-server", *maybe_redis_conf, "--port", str(port)])

        # give redis-server time to start
        # TODO this can be improved (also below) by checking whether the
        #  respective processes are in their expected states
        sleep(1)

        # initiate worker processes
        maybe_password = [] if password is None else ["--password", password]
        maybe_daemon = [] if daemon is None else ["--daemon", str(daemon)]
        self.workers = [
            Process(target=work,
                    args=(["--host", "localhost",
                           "--port", str(port),
                           *maybe_password,
                           "--processes", str(processes_per_worker),
                           *maybe_daemon,
                           "--catch", str(catch)],),
                    daemon=False)
            for _ in range(workers)
        ]

        # start workers
        for p in self.workers:
            p.start()

        # sleep a short amount of time to make sure everything is set up
        time.sleep(0.5)

    def shutdown(self):
        """Cleanup workers and server."""
        if not self.redis_server:
            # no server: stop() was likely called already, skip
            return

        # send stop signal to workers
        _manage("stop", port=self.port, password=self.password)
        for p in self.workers:
            # wait for workers to join
            p.join()

        # terminate server
        self.redis_server.terminate()
        # make sure it's gone
        self.redis_server.kill()
        # delete python reference
        del self.redis_server
        self.redis_server = None


class RedisEvalParallelSamplerServerStarter(RedisEvalParallelSampler):
    """
    Simple routine to start a dynamic redis-server.
    For the arguments see the base class.
    """

    def __init__(self,
                 password: str = None,
                 batch_size: int = 1,
                 wait_for_all_samples: bool = False,
                 look_ahead: bool = False,
                 look_ahead_delay_evaluation=True,
                 max_n_eval_look_ahead_factor: float = 10.,
                 workers: int = 2,
                 processes_per_worker: int = 1,
                 daemon: bool = True,
                 catch: bool = True,
                 log_file: str = None,
                 ):
        self.server_starter = RedisServerStarter(
            password=password, workers=workers,
            processes_per_worker=processes_per_worker,
            daemon=daemon, catch=catch)

        super().__init__(
            host="localhost", port=self.server_starter.port,
            password=self.server_starter.password,
            batch_size=batch_size, wait_for_all_samples=wait_for_all_samples,
            look_ahead=look_ahead,
            look_ahead_delay_evaluation=look_ahead_delay_evaluation,
            max_n_eval_look_ahead_factor=max_n_eval_look_ahead_factor,
            log_file=log_file)

    def shutdown(self):
        self.server_starter.shutdown()


class RedisStaticSamplerServerStarter(RedisStaticSampler):
    """
    Simple routine to start a static redis-server.
    For the arguments see the base class.
    """

    def __init__(self,
                 password: str = None,
                 workers: int = 2,
                 processes_per_worker: int = 1,
                 daemon: bool = True,
                 catch: bool = True,
                 log_file: str = None,
                 ):
        self.server_starter = RedisServerStarter(
            password=password, workers=workers,
            processes_per_worker=processes_per_worker,
            daemon=daemon, catch=catch)

        super().__init__(
            host="localhost", port=self.server_starter.port,
            password=self.server_starter.password, log_file=log_file)

    def shutdown(self):
        self.server_starter.shutdown()
