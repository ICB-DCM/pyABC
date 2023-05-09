"""
.. _api_weighted_statistics:

Weighted statistics
===================

Functions performing statistical operations on weighted points
generated via importance sampling.
"""

from .weighted_statistics import (
    effective_sample_size,
    resample,
    resample_deterministic,
    weighted_mean,
    weighted_median,
    weighted_mse,
    weighted_quantile,
    weighted_rmse,
    weighted_std,
    weighted_var,
)
