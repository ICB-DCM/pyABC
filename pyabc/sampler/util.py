"""Sampling util functions."""

from .base import Sample


def any_particle_preliminary(sample: Sample) -> bool:
    """Determine whether any particle in that sample is preliminary."""
    return any(particle.preliminary for particle in sample.particles)
