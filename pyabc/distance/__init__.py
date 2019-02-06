"""
Distance functions
==================

Commonly used distance functions for ABC, implementing a few advanced
features.
"""

from .base import(
    Distance,
    NoDistance,
    SimpleFunctionDistance,
    to_distance)

from .distance import (
    PNormDistance,
    AdaptivePNormDistance,
    ZScoreDistance,
    PCADistance,
    MinMaxDistance,
    PercentileDistance,
    RangeEstimatorDistance,
    DistanceWithMeasureList)

from .scales import (
    median_absolute_deviation,
    mean_absolute_deviation,
    standard_deviation,
    bias,
    root_mean_square_deviation,
    median_absolute_deviation_to_observation,
    mean_absolute_deviation_to_observation,
    combined_median_absolute_deviation,
    combined_mean_absolute_deviation,
    standard_deviation_to_observation)

from .kernel import (
    StochasticKernel,
    RET_SCALE_LIN,
    RET_SCALE_LOG,
    SimpleFunctionKernel,
    NormalKernel,
    IndependentNormalKernel,
    BinomialKernel)

__all__ = [
    # base
    "Distance",
    "NoDistance",
    "SimpleFunctionDistance",
    "to_distance",
    # distances
    "PNormDistance",
    "AdaptivePNormDistance",
    "ZScoreDistance",
    "PCADistance",
    "MinMaxDistance",
    "PercentileDistance",
    "RangeEstimatorDistance",
    "DistanceWithMeasureList",
    "to_distance",
    # scales
    "median_absolute_deviation",
    "mean_absolute_deviation",
    "standard_deviation",
    "bias",
    "root_mean_square_deviation",
    "median_absolute_deviation_to_observation",
    "mean_absolute_deviation_to_observation",
    "combined_median_absolute_deviation",
    "combined_mean_absolute_deviation",
    "standard_deviation_to_observation",
    # kernels
    "StochasticKernel",
    "RET_SCALE_LIN",
    "RET_SCALE_LOG",
    "SimpleFunctionKernel",
    "NormalKernel",
    "IndependentNormalKernel",
    "BinomialKernel",
]
