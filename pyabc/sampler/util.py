"""Sampling util functions."""

import multiprocessing as mp

from ..population import Sample


def any_particle_preliminary(sample: Sample) -> bool:
    """Determine whether any particle in that sample is preliminary."""
    return any(particle.preliminary for particle in sample.all_particles)


def get_mp_context():
    """Get a multiprocessing context.

    On POSIX, prefer ``fork`` when available to support non-picklable
    callables (e.g. local functions in tests) in nested worker setups.

    Queues, Values and Processes that are shared with each other must all be
    created from the *same* context. Mixing contexts (e.g. a default-context
    ``spawn`` queue with a ``fork`` process) can crash at runtime, so callers
    should obtain a single context here and derive all of them from it.
    """
    if 'fork' in mp.get_all_start_methods():
        return mp.get_context('fork')
    return mp.get_context()


def get_mp_process():
    """Get a multiprocessing Process constructor.

    On POSIX, prefer ``fork`` when available to support non-picklable
    callables (e.g. local functions in tests) in nested worker setups.
    """
    return get_mp_context().Process
