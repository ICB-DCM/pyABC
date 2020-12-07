from .base import Sample


def any_particle_preliminary(sample: Sample) -> bool:
    return any(particle.preliminary for particle in sample.particles)
