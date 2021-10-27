"""Util functions for tests."""

import os
import logging

# maximum population size environment variable
PYABC_MAX_POP_SIZE = "PYABC_MAX_POP_SIZE"

logger = logging.getLogger("ABC.Util")


def bound_pop_size_from_env(pop_size: int):
    """Bound population size if corresponding environment variable set.

    Parameters
    ----------
    pop_size: Intended population size

    Returns
    -------
    bounded_pop_size:
        Minimum of `pop_size` and environment variable `PYABC_MAX_POP_SIZE`.
    """
    if PYABC_MAX_POP_SIZE not in os.environ:
        return pop_size
    pop_size = min(pop_size, int(os.environ[PYABC_MAX_POP_SIZE]))

    logger.warning(
        f"Bounding population size to {pop_size} via environment variable "
        f"{PYABC_MAX_POP_SIZE}")

    return pop_size
