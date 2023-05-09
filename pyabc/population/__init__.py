"""
Particles and Populations
=========================

A particle contains the sampled parameters and simulated data.
A population gathers all particles collected in one SMC
iteration.
"""

from .population import Particle, Population, Sample, SampleFactory
