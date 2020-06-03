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
    Distance,
    NoDistance,
    IdentityFakeDistance,
    AcceptAllDistance,
    SimpleFunctionDistance,
    to_distance,
)
from .distance import (
    PNormDistance,
    AdaptivePNormDistance,
    AggregatedDistance,
    AdaptiveAggregatedDistance,
    ZScoreDistance,
    PCADistance,
    MinMaxDistance,
    PercentileDistance,
    RangeEstimatorDistance,
    DistanceWithMeasureList,
)
from .scale import (
    median_absolute_deviation,
    mean_absolute_deviation,
    standard_deviation,
    bias,
    root_mean_square_deviation,
    median_absolute_deviation_to_observation,
    mean_absolute_deviation_to_observation,
    combined_median_absolute_deviation,
    combined_mean_absolute_deviation,
    standard_deviation_to_observation,
    span,
    mean,
    median,
)
from .kernel import (
    StochasticKernel,
    SCALE_LIN,
    SCALE_LOG,
    SimpleFunctionKernel,
    NormalKernel,
    IndependentNormalKernel,
    IndependentLaplaceKernel,
    BinomialKernel,
    PoissonKernel,
    NegativeBinomialKernel,
)
