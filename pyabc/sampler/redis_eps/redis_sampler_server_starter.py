from time import sleep
from subprocess import Popen
from multiprocessing import Process
import psutil
from .cli import work, _manage
from .sampler import RedisEvalParallelSampler


class RedisEvalParallelSamplerServerStarter(RedisEvalParallelSampler):
    """
    Simple routine to start a redis-server with 2 workers for test purposes.
    """

    def __init__(self, host="localhost", port=6379, batch_size=1):
        # start server
        conn = psutil.net_connections()
        ports = [c.laddr[1] for c in conn]
        port = max(ports) + 1
        self.__port = port
        self.__redis_server = Popen(["redis-server", "--port", str(port)])

        # give redis-server time to start
        sleep(1)

        super().__init__(host, port, batch_size=batch_size)

        # initiate worker processes
        self.__worker = [
            Process(target=work,
                    args=(["--host", "localhost", "--port", str(port)],),
                    daemon=False)
            for _ in range(2)
        ]

        # start workers
        for p in self.__worker:
            p.start()

    def cleanup(self):
        """
        Cleanup workers and server.
        """
        # send stop signal to workers
        _manage("stop", port=self.__port)
        for p in self.__worker:
            # wait for workers to join
            p.join()
        # terminate server
        self.__redis_server.terminate()
        # make sure it's gone
        self.__redis_server.kill()
        # delete python reference
        del self.__redis_server
