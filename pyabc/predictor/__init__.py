"""
Predictor
=========

Predictor models are used in pyABC to regress parameters from data.
:class:`pypesto.predictor.Predictor` defines the abstract
base class, :class:`pypesto.predictor.SimplePredictor` an interface to external
predictor implementations.
Further, various specific implementations including linear regression, Lasso,
Gaussian processes, and neural networks are provided.
"""

from .predictor import (
    Predictor,
    SimplePredictor,
    LinearPredictor,
    LassoPredictor,
    GPPredictor,
    GPKernelHandle,
    MLPPredictor,
    HiddenLayerHandle,
    ModelSelectionPredictor,
    root_mean_square_error,
    root_mean_square_relative_error,
)
