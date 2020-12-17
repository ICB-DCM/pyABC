"""
External simulators
===================

This module can be used to easily interface pyabc with model simulations,
summary statistics calculators and distance functions written in arbitrary
programing languages, only requiring a specified command line interface
and file input and output.
"""

from .base import (
    ExternalDistance,
    ExternalHandler,
    ExternalModel,
    ExternalSumStat,
    create_sum_stat,
)
from .r_rpy2 import R
