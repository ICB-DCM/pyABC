"""
External simulators
===================

This module can be used to easily interface pyABC with model simulations,
summary statistics calculators and distance functions written in programming
languages other than Python.

The class :class:`pyabc.external.ExternalHandler`, as well as derived
Model, SumStat, and Distance classes, allow the use of arbitrary languages,
with communication via file i/o.
It has been successfully used with models written in e.g. R, Java, or C++.

Further, pyABC provides efficient interfaces to R via the class
:class:`pyabc.external.r.R` via the rpy2 package, and to Julia via the class
:class:`pyabc.external.julia.Julia` via the pyjulia package.
"""

from .base import (
    LOC,
    RETURNCODE,
    TIMEOUT,
    ExternalDistance,
    ExternalHandler,
    ExternalModel,
    ExternalSumStat,
    create_sum_stat,
)
