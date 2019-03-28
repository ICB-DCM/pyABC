import configparser
import os


DEFAULT_CONFIG = """#~/.parallel
[DIRECTORIES]
TMP=/tmp

[BROKER]
# can be SQLITE or REDIS
TYPE=REDIS

[SGE]
QUEUE=p.openmp
PARALLEL_ENVIRONMENT=openmp
PRIORITY=-500

[REDIS]
HOST=127.0.0.1
"""


def get_config():
    config = configparser.ConfigParser()
    config.read_string(DEFAULT_CONFIG)

    config_file = os.path.join(os.getenv("HOME"), ".parallel")
    if os.path.isfile(config_file):
        config.read(config_file)

    return config
