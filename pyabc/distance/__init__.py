"""
Distances
=========

Distance functions or metrics measure closeness of observed and sampled data.
This module implements various commonly used distance functions for ABC,
featuring a few advanced concepts.

For custom distance functions, either pass a plain function to ABCSMC, or
subclass the pyabc.Distance class.
"""

from .base import (
    Distance,
    NoDistance,
    AcceptAllDistance,
    SimpleFunctionDistance,
    to_distance,
)
from .distance import (
    ZScoreDistance,
    PCADistance,
    MinMaxDistance,
    PercentileDistance,
    RangeEstimatorDistance,
    DistanceWithMeasureList,
)
from .pnorm import (
    PNormDistance,
    AdaptivePNormDistance,
    InfoWeightedPNormDistance,
)
from .aggregate import (
    AggregatedDistance,
    AdaptiveAggregatedDistance,
)
from .ot import (
    WassersteinDistance,
    SlicedWassersteinDistance,
)
from .scale import (
    median_absolute_deviation,
    mad,
    mean_absolute_deviation,
    standard_deviation,
    std,
    bias,
    root_mean_square_deviation,
    rmsd,
    std_or_rmsd,
    median_absolute_deviation_to_observation,
    mado,
    mad_or_cmad,
    pcmad,
    mean_absolute_deviation_to_observation,
    combined_median_absolute_deviation,
    cmad,
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
