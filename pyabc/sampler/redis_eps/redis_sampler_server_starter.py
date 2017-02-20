from time import sleep
from subprocess import Popen
from multiprocessing import Process
import psutil
from .cli import work, _manage
from .sampler import RedisEvalParallelSampler


class RedisEvalParallelSamplerServerStarter(RedisEvalParallelSampler):
    def __init__(self, host="localhost", port=6379):
        conn = psutil.net_connections()
        ports = [c.laddr[1] for c in conn]
        port = max(ports) + 1
        self.__port = port
        self.__redis_server = Popen(["redis-server", "--port", str(port)])
        sleep(1)
        self.__worker = [
            Process(target=work,
                    args=(["--host", "localhost", "--port", str(port)],),
                    daemon=True)
            for _ in range(2)
        ]
        for p in self.__worker:
            p.start()
        super().__init__(host, port)

    def cleanup(self):
        _manage("stop", port=self.__port)
        for p in self.__worker:
            p.join()
        self.__redis_server.terminate()
        self.__redis_server.kill()
        del self.__redis_server
