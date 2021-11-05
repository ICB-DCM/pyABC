"""
Distances
=========

Distance functions or metrics measure closeness of observed and sampled data.
This module implements various commonly used distance functions for ABC,
featuring a few advanced concepts.

For custom distance functions, either pass a plain function to ABCSMC, or
subclass the pyabc.Distance class.
"""

from .aggregate import AdaptiveAggregatedDistance, AggregatedDistance
from .base import AcceptAllDistance, Distance, FunctionDistance, NoDistance
from .distance import (
    DistanceWithMeasureList,
    MinMaxDistance,
    PCADistance,
    PercentileDistance,
    RangeEstimatorDistance,
    ZScoreDistance,
)
from .kernel import (
    SCALE_LIN,
    SCALE_LOG,
    BinomialKernel,
    FunctionKernel,
    IndependentLaplaceKernel,
    IndependentNormalKernel,
    NegativeBinomialKernel,
    NormalKernel,
    PoissonKernel,
    StochasticKernel,
)
from .ot import SlicedWassersteinDistance, WassersteinDistance
from .pnorm import (
    AdaptivePNormDistance,
    InfoWeightedPNormDistance,
    PNormDistance,
)
from .scale import (
    bias,
    cmad,
    combined_mean_absolute_deviation,
    combined_median_absolute_deviation,
    mad,
    mad_or_cmad,
    mado,
    mean,
    mean_absolute_deviation,
    mean_absolute_deviation_to_observation,
    median,
    median_absolute_deviation,
    median_absolute_deviation_to_observation,
    pcmad,
    rmsd,
    root_mean_square_deviation,
    span,
    standard_deviation,
    standard_deviation_to_observation,
    std,
    std_or_rmsd,
)
