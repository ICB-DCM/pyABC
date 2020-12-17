"""
Distance functions
==================

Distance functions measure closeness of observed and sampled data. This
module implements various commonly used distance functions for ABC, featuring
a few advanced concepts.

For custom distance functions, either pass a plain function to ABCSMC or
subclass the pyabc.Distance class.
"""

from .base import (
    AcceptAllDistance,
    Distance,
    IdentityFakeDistance,
    NoDistance,
    SimpleFunctionDistance,
    to_distance,
)
from .distance import (
    AdaptiveAggregatedDistance,
    AdaptivePNormDistance,
    AggregatedDistance,
    DistanceWithMeasureList,
    MinMaxDistance,
    PCADistance,
    PercentileDistance,
    PNormDistance,
    RangeEstimatorDistance,
    ZScoreDistance,
)
from .kernel import (
    SCALE_LIN,
    SCALE_LOG,
    BinomialKernel,
    IndependentLaplaceKernel,
    IndependentNormalKernel,
    NegativeBinomialKernel,
    NormalKernel,
    PoissonKernel,
    SimpleFunctionKernel,
    StochasticKernel,
)
from .scale import (
    bias,
    combined_mean_absolute_deviation,
    combined_median_absolute_deviation,
    mean,
    mean_absolute_deviation,
    mean_absolute_deviation_to_observation,
    median,
    median_absolute_deviation,
    median_absolute_deviation_to_observation,
    root_mean_square_deviation,
    span,
    standard_deviation,
    standard_deviation_to_observation,
)
