"""
Summary statistics
==================

Summary statistics generally yield a lower-dimensional informative
representation of the model output. Distance comparisons are then performed in
summary statistics space.

The `pyabc.sumstat.Sumstat` base class allows to chain statistics, and to use
use self-learned and adaptive statistics. It is directly integrated in
distance functions.

.. note::
   Besides this summary statistics class integrated in the distance
   calculation, e.g. ABCSMC allows to specify `models` and
   `summary_statistics`. It is the output of `summary_statistics(model(...))`
   that is saved in the database. However, in general it does not have to be
   the final summary statistics, which are given by this module here.
   We acknowledge that the naming (due to legacy) may be confusing. The
   different layers all make sense, as they allow to separately specify what
   the model does, what information is to be saved, and on what representation
   to calculate distances.
"""

from .base import (
    Sumstat,
    IdentitySumstat,
)
from .learn import (
    PredictorSumstat,
    LinearPredictorSumstat,
    GPPredictorSumstat,
)