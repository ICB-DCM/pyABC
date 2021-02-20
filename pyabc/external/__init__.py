"""
External simulators
===================

This module can be used to easily interface pyABC with model simulations,
summary statistics calculators and distance functions written in arbitrary
programming languages, only requiring a specified command line interface
and file input and output.
It has been successfully used with models written in e.g. R, Java, or C++.
"""

from .r_rpy2 import R
from .base import (
    ExternalHandler,
    ExternalModel,
    ExternalSumStat,
    ExternalDistance,
    create_sum_stat)
