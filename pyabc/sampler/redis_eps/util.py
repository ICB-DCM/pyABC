import signal
import sys
import cloudpickle as pickle
from redis import StrictRedis

from .cmd import ACTIVE_SET, ACTIVE_SET_LOCK, idfy
from .redis_logging import logger


class KillHandler:
    """Handle killing workers during delicate processes."""
    def __init__(self):
        self.killed = False
        self.exit = True
        signal.signal(signal.SIGTERM, self.handle)
        signal.signal(signal.SIGINT, self.handle)

    def handle(self, *args):
        self.killed = True
        if self.exit:
            sys.exit(0)


def get_active_set(redis: StrictRedis, ana_id: str, t: int) -> set:
    """Read active set from redis."""
    active_set = redis.get(idfy(ACTIVE_SET, ana_id, t))
    if active_set is not None:
        active_set = pickle.loads(active_set)
    return active_set


def set_active_set(
    redis: StrictRedis, ana_id: str, t: int, active_set: set,
) -> None:
    """Store active set to redis."""
    redis.set(idfy(ACTIVE_SET, ana_id, t), pickle.dumps(active_set))


def discard_ix_from_active_set(
    redis: StrictRedis, ana_id: str, t: int, ix: int,
) -> None:
    """Discard an entry from active set."""
    with redis.lock(ACTIVE_SET_LOCK):
        active_set: set = get_active_set(redis=redis, ana_id=ana_id, t=t)
        if active_set is None:
            logger.info("Could not discard from active set: Doesn't exist.")
            return
        active_set.discard(ix)
        set_active_set(redis=redis, ana_id=ana_id, t=t, active_set=active_set)


def add_ix_to_active_set(
    redis: StrictRedis, ana_id: str, t: int, ix: int,
) -> None:
    """Add entry to active set."""
    with redis.lock(ACTIVE_SET_LOCK):
        active_set: set = get_active_set(redis=redis, ana_id=ana_id, t=t)
        if active_set is None:
            logger.info("Could not add to active set: Doesn't exist.")
            return
        active_set.add(ix)
        set_active_set(redis=redis, ana_id=ana_id, t=t, active_set=active_set)
