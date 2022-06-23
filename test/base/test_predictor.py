"""Tests for the `pyabc.predictor` module."""

import numpy as np
import pytest

from pyabc.predictor import (
    GPKernelHandle,
    GPPredictor,
    HiddenLayerHandle,
    LassoPredictor,
    LinearPredictor,
    MLPPredictor,
    ModelSelectionPredictor,
    root_mean_square_error,
    root_mean_square_relative_error,
)


@pytest.fixture(
    params=[
        "linear",
        "linear not joint",
        "linear not normalized",
        "linear weighted",
        "linear not joint weighted",
        "lasso",
        "GP no ard",
        "GP no ard not joint",
        "GP ard",
        "MLP heuristic",
        "MLP mean",
        "MLP max",
        "MS TTS",
        "MS CV",
    ]
)
def s_predictor(request) -> str:
    return request.param


@pytest.fixture(params=["linear", "quadratic"])
def s_model(request) -> str:
    return request.param


@pytest.mark.flaky(reruns=5)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_fit(s_model, s_predictor):
    """Test fit on a simple model."""

    n_y = 10
    n_p = 3
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
            return np.dot(y, m) ** 2 + b

    else:
        raise ValueError("Invalid argument")

    if s_predictor == "linear":
        predictor = LinearPredictor()
    elif s_predictor == "linear not joint":
        predictor = LinearPredictor(joint=False)
    elif s_predictor == "linear not normalized":
        predictor = LinearPredictor(
            normalize_features=False, normalize_labels=False
        )
    elif s_predictor == "linear weighted":
        predictor = LinearPredictor(weight_samples=True)
    elif s_predictor == "linear not joint weighted":
        predictor = LinearPredictor(joint=False, weight_samples=True)
    elif s_predictor == "lasso":
        predictor = LassoPredictor()
    elif s_predictor == "GP no ard":
        predictor = GPPredictor(kernel=GPKernelHandle(ard=False))
    elif s_predictor == "GP no ard not joint":
        predictor = GPPredictor(kernel=GPKernelHandle(ard=False), joint=False)
    elif s_predictor == "GP ard":
        predictor = GPPredictor()
    elif s_predictor == "MLP heuristic":
        predictor = MLPPredictor(
            hidden_layer_sizes=HiddenLayerHandle(method="heuristic")
        )
    elif s_predictor == "MLP mean":
        predictor = MLPPredictor(
            hidden_layer_sizes=HiddenLayerHandle(method="mean")
        )
    elif s_predictor == "MLP max":
        predictor = MLPPredictor(
            hidden_layer_sizes=HiddenLayerHandle(method="max")
        )
    elif s_predictor == "MS TTS":
        predictor = ModelSelectionPredictor(
            predictors=[LinearPredictor(), MLPPredictor()],
            split_method="train_test_split",
        )
    elif s_predictor == "MS CV":
        predictor = ModelSelectionPredictor(
            predictors=[LinearPredictor(), MLPPredictor()],
            split_method="cross_validation",
        )
    else:
        raise ValueError("Invalid argument")

    # training data
    ys_train = rng.normal(size=(n_sample_train, n_y))
    ps_train = model(ys_train)
    w_train = 1 + 0.01 * rng.normal(size=(n_sample_train,))
    assert ps_train.shape == (n_sample_train, n_p)

    # fit model
    predictor.fit(x=ys_train, y=ps_train, w=w_train)

    # test data
    ys_test = rng.normal(size=(n_sample_test, n_y))
    ps_test = model(ys_test)
    ps_pred = predictor.predict(ys_test)
    assert ps_pred.shape == ps_test.shape == (n_sample_test, n_p)

    # measure discrepancy
    rmse = root_mean_square_error(ps_test, ps_pred, 1.0)
    print(f"rmse {s_predictor} {s_model}:", rmse)

    # ignore lasso
    if isinstance(
        predictor,
        (LinearPredictor, GPPredictor, MLPPredictor, ModelSelectionPredictor),
    ):
        if s_model == "linear":
            if s_predictor not in ["linear not normalized"]:
                # all should work well on linear
                assert rmse < 0.1
        elif not isinstance(predictor, LinearPredictor):
            # none is overwhelming on quadratic
            assert rmse < 10
        else:
            # linear model cannot fit non-linear data
            assert rmse > 10

    # call on a single vector
    assert predictor.predict(rng.normal(size=n_y)).shape == (1, n_p)

    # for visual analysis
    # import matplotlib.pyplot as plt
    # plt.plot(ps_test, ps_pred, '.')
    # plt.show()


def test_error_functions():
    """Test error functions used in model selection."""
    rng = np.random.Generator(np.random.PCG64(0))
    ys = rng.normal(size=(10, 100))
    assert root_mean_square_error(ys, ys, 1.0) == 0.0
    assert root_mean_square_error(ys, ys - 1, 1.0) == 1.0
    assert root_mean_square_relative_error(ys, ys) == 0.0
    assert root_mean_square_relative_error(ys, ys - 1) > 0.0


def test_wrong_input():
    """Test all kinds of wrong inputs."""
    with pytest.raises(ValueError):
        HiddenLayerHandle(method="potato")(n_in=10, n_out=10, n_sample=100)
