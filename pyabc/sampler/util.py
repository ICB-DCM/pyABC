"""Sampling util functions."""

import multiprocessing as mp

from ..population import Sample


def any_particle_preliminary(sample: Sample) -> bool:
    """Determine whether any particle in that sample is preliminary."""
    return any(particle.preliminary for particle in sample.all_particles)


def get_mp_process():
    """Get a multiprocessing Process constructor.

    On POSIX, prefer ``fork`` when available to support non-picklable
    callables (e.g. local functions in tests) in nested worker setups.
    """
    if 'fork' in mp.get_all_start_methods():
        return mp.get_context('fork').Process
    return mp.Process
