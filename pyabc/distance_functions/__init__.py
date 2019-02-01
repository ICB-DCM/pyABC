"""
Distance functions
==================
"""

from .distance_functions import (DistanceFunction,
                                 NoDistance,
                                 SimpleFunctionDistance,
                                 PNormDistance,
                                 AdaptivePNormDistance,
                                 ZScoreDistanceFunction,
                                 PCADistanceFunction,
                                 MinMaxDistanceFunction,
                                 PercentileDistanceFunction,
                                 RangeEstimatorDistanceFunction,
                                 DistanceFunctionWithMeasureList,
                                 to_distance)

from .scales import (median_absolute_deviation,
                     mean_absolute_deviation,
                     standard_deviation,
                     bias,
                     root_mean_square_deviation,
                     median_absolute_deviation_to_observation,
                     mean_absolute_deviation_to_observation,
                     combined_median_absolute_deviation,
                     combined_mean_absolute_deviation,
                     standard_deviation_to_observation)

__all__ = [
    # distances
    "DistanceFunction",
    "NoDistance",
    "SimpleFunctionDistance",
    "PNormDistance",
    "AdaptivePNormDistance",
    "ZScoreDistanceFunction",
    "PCADistanceFunction",
    "MinMaxDistanceFunction",
    "PercentileDistanceFunction",
    "RangeEstimatorDistanceFunction",
    "DistanceFunctionWithMeasureList",
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
    "standard_deviation_to_observation"
]
