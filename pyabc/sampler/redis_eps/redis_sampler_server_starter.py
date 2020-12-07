from time import sleep
from subprocess import Popen  # noqa: S404
from multiprocessing import Process
import tempfile
import psutil
import time

from .cli import work, _manage
from .sampler import RedisEvalParallelSampler


class RedisEvalParallelSamplerServerStarter(RedisEvalParallelSampler):
    """
    Simple routine to start a redis-server with 2 workers for test purposes.
    For the arguments see the base class.
    """

    def __init__(self,
                 host: str = "localhost",
                 password: str = None,
                 batch_size: int = 1,
                 look_ahead: bool = False,
                 look_ahead_delay_evaluation=True,
                 workers: int = 2,
                 processes_per_worker: int = 1,
                 catch: bool = True,
                 log_file: str = None,
                 ):
        # start server
        conn = psutil.net_connections()
        ports = [c.laddr[1] for c in conn]
        port = max(ports) + 1
        self.__port = port
        self.__password = password

        # create config file
        maybe_redis_conf = []
        if password is not None:
            fname = tempfile.mkstemp()[1]
            with open(fname, 'w') as f:
                f.write(f"requirepass {password}\n")
            maybe_redis_conf = [fname]

        self.__redis_server = Popen(  # noqa: S607,S603
            ["redis-server", *maybe_redis_conf, "--port", str(port)])

        # give redis-server time to start
        # TODO this can be improved (also below) by checking whether the
        #  respective processes are in their expected states
        sleep(1)

        super().__init__(
            host, port, password, batch_size=batch_size, look_ahead=look_ahead,
            look_ahead_delay_evaluation=look_ahead_delay_evaluation,
            log_file=log_file)

        # initiate worker processes
        maybe_password = [] if password is None else ["--password", password]
        self.__worker = [
            Process(target=work,
                    args=(["--host", "localhost",
                           "--port", str(port),
                           *maybe_password,
                           "--processes", str(processes_per_worker),
                           "--catch", str(catch)],),
                    daemon=False)
            for _ in range(workers)
        ]

        # start workers
        for p in self.__worker:
            p.start()

        # sleep a short amount of time to make sure everything is set up
        time.sleep(0.5)

    def shutdown(self):
        """Cleanup workers and server."""
        # call stop command
        self.stop()
        if not self.__redis_server:
            # no server: stop() was likely called already, skip
            return

        # send stop signal to workers
        _manage("stop", port=self.__port, password=self.__password)
        for p in self.__worker:
            # wait for workers to join
            p.join()

        # terminate server
        self.__redis_server.terminate()
        # make sure it's gone
        self.__redis_server.kill()
        # delete python reference
        del self.__redis_server
        self.__redis_server = None
