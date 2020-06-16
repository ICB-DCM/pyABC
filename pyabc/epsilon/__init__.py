"""
Epsilons
========

Epsilon threshold updating strategies.

Acceptance thresholds (= epsilon) can be calculated based on the distances from
the observed data, can follow a pre-defined list, can be constant, or can have
a user-defined implementation.
"""


from .base import (
    Epsilon,
    NoEpsilon,
)
from .epsilon import (
    ConstantEpsilon,
    ListEpsilon,
    QuantileEpsilon,
    MedianEpsilon,
)
from .temperature import (
    TemperatureBase,
    ListTemperature,
    Temperature,
    TemperatureScheme,
    AcceptanceRateScheme,
    ExpDecayFixedIterScheme,
    ExpDecayFixedRatioScheme,
    PolynomialDecayFixedIterScheme,
    DalyScheme,
    FrielPettittScheme,
    EssScheme,
)
