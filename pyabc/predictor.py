import numpy as np
from typing import List, Union
import copy
import logging
from abc import ABC, abstractmethod

try:
    import sklearn.linear_model as skl_lm
    import sklearn.gaussian_process as skl_gp
    import sklearn.neural_network as skl_nn
    import sklearn.model_selection as skl_ms
except ImportError as e:
    skl_lm = skl_gp = skl_nn = skl_ms = None


logger = logging.getLogger("Predictor")


class Predictor(ABC):
    """Generic predictor model class.

    A predictor should define:

    - `fit(x, y, w=None)` to fit the model on a sample of data x and outputs y,
      where x has shape (n_sample, n_feature), and
      y has shape (n_sample, n_out).
      Further, gets as a third argument the sample weights
      if `weight_samples` is set. Not all predictors support this.
    - `predict(X)` to predict outputs of shape (n_out,), where X has shape
      (n_sample, n_feature).
    """

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> None:
        """Fit the predictor to labeled data.

        Parameters
        ----------
        x: Samples, shape (n_sample, n_feature).
        y: Targets, shape (n_sample, n_out).
        w: Weights, shape (n_sample,).
        """

    @abstractmethod
    def predict(self, x: np.ndarray, orig_scale: bool = True) -> np.ndarray:
        """Predict outputs using the model.

        Parameters
        ---------
        x: Samples, shape (n_sample, n_feature) or (n_feature,).
        orig_scale: Whether outputs should be on the original scale.

        Returns
        -------
        y: Predicted targets, shape (n_sample, n_out).
        """


class SimplePredictor(Predictor):
    """Wrapper around generic predictor routines."""

    def __init__(
        self,
        predictor,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        joint: bool = True,
        weight_samples: bool = False,
    ):
        """
        Parameters
        -----------
        predictor:
            Predictor model to use, fulfilling the predictor contract.
        normalize_features:
            Whether to apply z-score normalization to the input data.
        normalize_labels:
            Whether to apply z-score normalization to the parameters.
        joint:
            Whether the predictor learns one model for all targets, or
            separate models per target.
        weight_samples:
            Whether to use importance sampling weights. Not that not all
            predictors may support weighted samples.
        """
        self.predictor = predictor
        # only used if not joint
        self.single_predictors: Union[List, None] = None

        self.normalize_features: bool = normalize_features
        self.normalize_labels: bool = normalize_labels

        self.joint: bool = joint
        self.weight_samples: bool = weight_samples

        # z-score normalization coefficients
        self.mean_x: Union[np.ndarray, None] = None
        self.std_x: Union[np.ndarray, None] = None
        self.mean_y: Union[np.ndarray, None] = None
        self.std_y: Union[np.ndarray, None] = None

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> None:
        """Fit the predictor to labeled data.

        Parameters
        ----------
        x: Samples, shape (n_sample, n_feature).
        y: Targets, shape (n_sample, n_out).
        w: Weights, shape (n_sample,).
        """
        # normalize features
        if self.normalize_features:
            self.mean_x = np.mean(x, axis=0)
            self.std_x = np.std(x, axis=0)
            x = (x - self.mean_x) / self.std_x

        # normalize labels
        if self.normalize_labels:
            self.mean_y = np.mean(y, axis=0)
            self.std_y = np.std(y, axis=0)
            y = (y - self.mean_y) / self.std_y

        if self.joint:
            # fit a model with all parameters as joint response variables
            if self.weight_samples:
                self.predictor.fit(x, y, w)
            else:
                self.predictor.fit(x, y)
        else:
            # set up predictors
            if self.single_predictors is None:
                n_par = x.shape[1]
                self.single_predictors: List = [
                    copy.deepcopy(self.predictor) for _ in range(n_par)]
            # fit a model for each parameter separately
            for predictor, y_ in zip(self.single_predictors, y.T):
                if self.weight_samples:
                    predictor.fit(x, y_, w)
                else:
                    predictor.fit(x, y_)

        logger.info(f"Fitted {self.__class__.__name__}")

    def predict(self, x: np.ndarray, orig_scale: bool = True) -> np.ndarray:
        """Predict outputs using the model.

        Parameters
        ---------
        x: Samples, shape (n_sample, n_feature) or (n_feature,).
        orig_scale: Whether outputs should be on the original scale.

        Returns
        -------
        y: Predicted targets, shape (n_sample, n_out).
        """
        # normalize features
        if self.normalize_features:
            x = (x - self.mean_x) / self.std_x

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if self.joint:
            y = self.predictor.predict(x)
        else:
            y = [predictor.predict(x) for predictor in self.single_predictors]
            y = np.array(y).T

        if orig_scale and self.normalize_labels:
            y = y * self.std_y + self.mean_y

        return y


class LinearPredictor(SimplePredictor):
    """Linear predictor model."""

    def __init__(
        self,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        joint: bool = True,
        weight_samples: bool = False,
    ):
        # check installation
        if skl_lm is None:
            raise ImportError(
                "This predictor requires an installation of scikit-learn. "
                "Install e.g. via `pip install pyabc[scikit-learn]`")

        predictor = skl_lm.LinearRegression(
            fit_intercept=False, normalize=False,
        )

        super().__init__(
            predictor=predictor,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            weight_samples=weight_samples,
        )


class LassoPredictor(SimplePredictor):
    """Lasso (least absolute shrinkage and selection) model.

    Linear model with l1 regularization.
    """

    def __init__(
        self,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        joint: bool = True,
        **kwargs,
    ):
        """Additional keyword arguments are passed on to the model."""
        # check installation
        if skl_lm is None:
            raise ImportError(
                "This predictor requires an installation of scikit-learn. "
                "Install e.g. via `pip install pyabc[scikit-learn]`")

        # translate arguments
        all_args = {
            'fit_intercept': False,
            'normalize': False,
        }
        all_args.update(kwargs)
        predictor = skl_lm.Lasso(**all_args)

        super().__init__(
            predictor=predictor,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            weight_samples=False,
        )


class GPPredictor(SimplePredictor):
    """Gaussian process model."""

    def __init__(
        self,
        kernel=None,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        joint: bool = True,
        **kwargs,
    ):
        # check installation
        if skl_gp is None:
            raise ImportError(
                "This predictor requires an installation of scikit-learn. "
                "Install e.g. via `pip install pyabc[scikit-learn]`")

        # default kernel
        if kernel is None:
            kernel = skl_gp.kernels.RBF() + skl_gp.kernels.WhiteKernel()

        predictor = skl_gp.GaussianProcessRegressor(kernel=kernel, **kwargs)

        super().__init__(
            predictor=predictor,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            weight_samples=False,
        )


class MLPPredictor(SimplePredictor):
    """Multi-layer perceptron regressor predictor."""

    def __init__(
            self,
            normalize_features: bool = True,
            normalize_labels: bool = True,
            joint: bool = True,
            **kwargs,
    ):
        """Additional keyword arguments are passed on to the model."""
        # check installation
        if skl_nn is None:
            raise ImportError(
                "This predictor requires an installation of scikit-learn. "
                "Install e.g. via `pip install pyabc[scikit-learn]`")

        self.kwargs = {
            'hidden_layer_sizes': None,
            'solver': 'lbfgs',
            'max_iter': 10000,
        }
        self.kwargs.update(kwargs)

        super().__init__(
            predictor=None,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            weight_samples=False,
        )

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> None:
        # need to recreate the model
        kwargs = copy.deepcopy(self.kwargs)

        # default hidden layer sizes
        if kwargs['hidden_layer_sizes'] is None:
            n_in, n_out = x.shape[1], y.shape[1]
            if not self.joint:
                n_out = 1
            hidden_layer_sizes = (
                int(n_in * 1.5),
                int(0.5 * (n_in + n_out)),
            )
            kwargs['hidden_layer_sizes'] = hidden_layer_sizes

        self.predictor = skl_nn.MLPRegressor(**kwargs)
        super().fit(x, y, w)


class ModelSelection(Predictor):
    """Model selection over a set of predictors.

    Performs k-fold cross validation with root mean square error score on
    z-score normalized targets.
    """

    def __init__(
        self,
        predictors: List[Predictor],
        n_splits: int = 5,
    ):
        """
        Parameters
        ----------
        predictors:
            Set of predictors over which to perform model selection.
        """
        super().__init__()
        self.predictors: List[Predictor] = predictors
        self.n_splits: int = n_splits

        # holds the chosen predictor model
        self.chosen_one: Union[Predictor, None] = None

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> None:
        # output normalization
        std_y = np.std(y, axis=0)

        # k-fold cross validation
        kfold = skl_ms.KFold(n_splits=self.n_splits, shuffle=True)
        scores = np.empty((len(self.predictors), self.n_splits))

        # iterate over cross validation sets
        for i_split, (train_ix, test_ix) in enumerate(
                kfold.split(np.arange(x.shape[0]))):
            # training and test sets
            x_train, y_train, w_train = x[train_ix], y[train_ix], w[train_ix]
            x_test, y_test = x[test_ix], y[test_ix]

            # iterate over predictors
            for i_predictor, predictor in enumerate(self.predictors):
                # fit predictor on training set
                predictor.fit(x=x_train, y=y_train, w=w_train)

                # get predictions on original scale
                y_predicted = predictor.predict(x=x_test, orig_scale=True)

                # score of z-score normalized variables
                scores[i_predictor, i_split] = np.sqrt(
                    np.sum(
                        ((y_predicted - y_test) / std_y)**2,
                    ) / x_train.shape[0]
                )

        # the score of a predictor is the sum over all folds
        scores = np.sum(scores, axis=1)

        # logging
        for predictor, score in zip(self.predictors, scores):
            logger.info(f"Score {predictor.__class__.__name__}: {score:.2e}")

        # best predictor has minimum score
        self.chosen_one = self.predictors[np.argmin(scores)]

        # retrain chosen model on full data set
        self.chosen_one.fit(x=x, y=y, w=w)

    def predict(self, x: np.ndarray, orig_scale: bool = True) -> np.ndarray:
        return self.chosen_one.predict(x=x, orig_scale=orig_scale)
