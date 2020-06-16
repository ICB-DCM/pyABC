"""
External simulators
===================

This module can be used to easily interface pyabc with model simulations,
summary statistics calculators and distance functions written in arbitrary
programing languages, only requiring a specified command line interface
and file input and output.
"""

from .r_rpy2 import R
from .base import (
    ExternalHandler,
    ExternalModel,
    ExternalSumStat,
    ExternalDistance,
    create_sum_stat)
