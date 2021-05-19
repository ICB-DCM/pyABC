"""Tests for the `pyabc.predictor` module."""

import pytest
import numpy as np

from pyabc.predictor import (
    LinearPredictor,
    LassoPredictor,
    GPPredictor,
    GPKernelHandle,
    MLPPredictor,
    HiddenLayerHandle,
    root_mean_square_error,
)


@pytest.fixture(params=[
    "linear",
    "linear not joint",
    "lasso",
    "GP no ard",
    "GP no ard not joint",
    "GP ard",
    "MLP heuristic",
    "MLP mean",
])
def s_predictor(request) -> str:
    return request.param


@pytest.fixture(params=["linear", "quadratic"])
def s_model(request) -> str:
    return request.param


def test_fit(s_model, s_predictor):
    """Test fit on a simple model."""
    n_y = 10
    n_p = 4
    n_sample_train = 500
    n_sample_test = 100

    rng = np.random.Generator(np.random.PCG64(0))

    m = 1 + rng.normal(size=(n_y, n_p))
    b = 1 + rng.normal(size=(1, n_p))

    if s_model == "linear":
        def model(y: np.ndarray) -> np.ndarray:
            return np.dot(y, m) + b
    elif s_model == "quadratic":
        def model(y: np.ndarray) -> np.ndarray:
            return np.dot(y, m)**2 + b
    else:
        raise ValueError("Invalid argument")

    if s_predictor == "linear":
        predictor = LinearPredictor()
    elif s_predictor == "linear not joint":
        predictor = LinearPredictor(joint=False)
    elif s_predictor == "lasso":
        predictor = LassoPredictor()
    elif s_predictor == "GP no ard":
        predictor = GPPredictor(kernel=GPKernelHandle(ard=False))
    elif s_predictor == "GP no ard not joint":
        predictor = GPPredictor(kernel=GPKernelHandle(ard=False), joint=False)
    elif s_predictor == "GP ard":
        predictor = GPPredictor(kernel=GPKernelHandle(ard=True))
    elif s_predictor == "MLP heuristic":
        predictor = MLPPredictor(
            hidden_layer_sizes=HiddenLayerHandle(method="heuristic"))
    elif s_predictor == "MLP mean":
        predictor = MLPPredictor(
            hidden_layer_sizes=HiddenLayerHandle(method="mean"))
    else:
        raise ValueError("Invalid argument")

    # training data
    ys_train = rng.normal(size=(n_sample_train, n_y))
    ps_train = model(ys_train)
    assert ps_train.shape == (n_sample_train, n_p)

    # fit model
    predictor.fit(x=ys_train, y=ps_train, w=None)

    # test data
    ys_test = rng.normal(size=(n_sample_test, n_y))
    ps_test = model(ys_test)
    ps_pred = predictor.predict(ys_test)

    # measure discrepancy
    rmse = root_mean_square_error(ps_test, ps_pred, 1.)
    print(f"rmse {s_predictor} {s_model}:", rmse)

    # ignore lasso
    if isinstance(
            predictor, (LinearPredictor, GPPredictor, MLPPredictor)):
        if s_model == "linear":
            # all should work well on linear
            assert rmse < 0.1
        elif not isinstance(predictor, LinearPredictor):
            # none is overwhelming on quadratic
            assert rmse < 10
        else:
            # linear model cannot fit non-linear data
            assert rmse > 10

    # for visual analysis
    # import matplotlib.pyplot as plt
    # plt.plot(ps_test, ps_pred, '.')
    # plt.show()
