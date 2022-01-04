"""
Epsilons
========

Epsilon threshold updating strategies.

Acceptance thresholds (= epsilon) can be calculated based on the distances from
the observed data, can follow a pre-defined list, can be constant, or can have
a user-defined implementation.
"""

from .base import Epsilon, NoEpsilon
from .epsilon import (
    ConstantEpsilon,
    ListEpsilon,
    MedianEpsilon,
    QuantileEpsilon,
)
from .silk import SilkOptimalEpsilon
from .temperature import (
    AcceptanceRateScheme,
    DalyScheme,
    EssScheme,
    ExpDecayFixedIterScheme,
    ExpDecayFixedRatioScheme,
    FrielPettittScheme,
    ListTemperature,
    PolynomialDecayFixedIterScheme,
    Temperature,
    TemperatureBase,
    TemperatureScheme,
)
