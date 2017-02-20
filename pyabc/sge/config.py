import configparser
import os


DEFAULT_CONFIG = """#~/.parallel
[DIRECTORIES]
TMP=/tmp

[BROKER]
TYPE=REDIS   # can be SQLITE or REDIS

[SGE]
QUEUE=p.openmp
PARALLEL_ENVIRONMENT=openmp
PRIORITY=-500

[REDIS]
HOST=127.0.0.1
"""


def make_config():
    config = configparser.ConfigParser()
    config.read_string(DEFAULT_CONFIG)
    config.read(os.path.join(os.getenv("HOME"), ".parallel"))
    return config
