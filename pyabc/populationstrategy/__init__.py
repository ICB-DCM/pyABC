"""
Population strategies
=====================

Strategies to choose the population size.

The population size can be constant or can change over the course
of the generations.
"""

from .populationstrategy import (
    AdaptivePopulationSize,
    ConstantPopulationSize,
    ListPopulationSize,
    PopulationStrategy,
)
