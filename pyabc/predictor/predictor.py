"""Predictor implementations."""

import numpy as np
from scipy.stats import pearsonr
from typing import Callable, List, Tuple, Union
import copy
import logging
from abc import ABC, abstractmethod
from time import time

try:
    import sklearn.linear_model as skl_lm
    import sklearn.gaussian_process as skl_gp
    import sklearn.neural_network as skl_nn
    import sklearn.model_selection as skl_ms
except ImportError:
    skl_lm = skl_gp = skl_nn = skl_ms = None


logger = logging.getLogger("ABC.Predictor")


class Predictor(ABC):
    """Generic predictor model class.

    A predictor should define:

    * `fit(x, y, w=None)` to fit the model on a sample of data x and outputs y,
      where x has shape (n_sample, n_feature), and
      y has shape (n_sample, n_out).
      Further, gets as a third argument the sample weights
      if `weight_samples` is set. Not all predictors support this.
    * `predict(X)` to predict outputs of shape (n_out,), where X has shape
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
    def predict(self, x: np.ndarray, normalize: bool = False) -> np.ndarray:
        """Predict outputs using the model.

        Parameters
        ----------
        x:
            Samples, shape (n_sample, n_feature) or (n_feature,).
        normalize:
            Whether outputs should be normalized, or on the original scale.

        Returns
        -------
        y: Predicted targets, shape (n_sample, n_out).
        """


def wrap_fit_log(fit):
    """Wrapper for fit logging."""
    def wrapped_fun(self, x: np.ndarray, y: np.ndarray, w: np.ndarray):
        start_time = time()

        # actual fitting
        ret = fit(self, x, y, w)

        logger.info(f"Fitted {self} in {time() - start_time:.2f}s")
        if self.log_pearson:
            # shape: n_sample, n_out
            y_pred = self.predict(x)
            coeffs = [
                pearsonr(y[:, i], y_pred[:, i])[0]
                for i in range(y_pred.shape[1])
            ]
            logger.info(" ".join([
                "Pearson correlations:",
                *[f"{coeff:.3f}" for coeff in coeffs]]),
            )

        return ret
    return wrapped_fun


class SimplePredictor(Predictor):
    """Wrapper around generic predictor routines."""

    def __init__(
        self,
        predictor,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        joint: bool = True,
        weight_samples: bool = False,
        log_pearson: bool = True,
    ):
        """
        Parameters
        ----------
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
        log_pearson:
            Whether to log Pearson correlation coefficients after fitting.
        """
        self.predictor = predictor
        # only used if not joint
        self.single_predictors: Union[List, None] = None

        self.normalize_features: bool = normalize_features
        self.normalize_labels: bool = normalize_labels

        self.joint: bool = joint
        self.weight_samples: bool = weight_samples

        self.log_pearson: bool = log_pearson

        # indices to use
        self.use_ixs: Union[np.ndarray, None] = None

        # z-score normalization coefficients
        self.mean_x: Union[np.ndarray, None] = None
        self.std_x: Union[np.ndarray, None] = None
        self.mean_y: Union[np.ndarray, None] = None
        self.std_y: Union[np.ndarray, None] = None

    @wrap_fit_log
    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> None:
        """Fit the predictor to labeled data.

        Parameters
        ----------
        x: Samples, shape (n_sample, n_feature).
        y: Targets, shape (n_sample, n_out).
        w: Weights, shape (n_sample,).
        """
        # remove trivial features
        self.set_use_ixs(x=x)
        x = x[:, self.use_ixs]

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
                n_par = y.shape[1]
                self.single_predictors: List = [
                    copy.deepcopy(self.predictor) for _ in range(n_par)]
            # fit a model for each parameter separately
            for predictor, y_ in zip(self.single_predictors, y.T):
                if self.weight_samples:
                    predictor.fit(x, y_, w)
                else:
                    predictor.fit(x, y_)

    def predict(self, x: np.ndarray, normalize: bool = False) -> np.ndarray:
        """Predict outputs using the model.

        Parameters
        ----------
        x:
            Samples, shape (n_sample, n_feature) or (n_feature,).
        normalize:
            Whether outputs should be normalized, or on the original scale.

        Returns
        -------
        y: Predicted targets, shape (n_sample, n_out).
        """
        # to 2d matrix
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # remove trivial features
        x = x[:, self.use_ixs]

        # normalize features
        if self.normalize_features:
            x = (x - self.mean_x) / self.std_x

        if self.joint:
            y = self.predictor.predict(x)
        else:
            y = [predictor.predict(x).flatten()
                 for predictor in self.single_predictors]
            y = np.array(y).T

        if not normalize and self.normalize_labels:
            y = y * self.std_y + self.mean_y

        # some predictors may return flat arrays if n_out == 1
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        return y

    def set_use_ixs(self, x: np.ndarray, log: bool = True) -> None:
        """Set feature indices to use.

        Parameters
        ----------
        x: Feature matrix, shape (n_sample, n_feature).
        log: Whether to log.
        """
        # remove trivial features
        self.use_ixs = np.any(x != x[0], axis=0)

        # log omitted indices
        if log and not self.use_ixs.all():
            logger.info(
                "Ignore trivial features "
                f"{list(np.flatnonzero(~self.use_ixs))}")

    def __repr__(self) -> str:
        rep = f"<{self.__class__.__name__} predictor={self.predictor}"
        # print everything that is customized
        if not self.normalize_features:
            rep += f" normalize_features={self.normalize_features}"
        if not self.normalize_labels:
            rep += f" normalize_labels={self.normalize_labels}"
        if not self.joint:
            rep += f" joint={self.joint}"
        if self.weight_samples:
            rep += f" weight_samples={self.weight_samples}"
        return rep + ">"


class LinearPredictor(SimplePredictor):
    """Linear predictor model.

    Based on [#fearnheadprangle2012]_.

    .. [#fearnheadprangle2012]
        Fearnhead, Paul, and Dennis Prangle.
        "Constructing summary statistics for approximate Bayesian computation:
        Semi‐automatic approximate Bayesian computation."
        Journal of the Royal Statistical Society: Series B
        (Statistical Methodology) 74.3 (2012): 419-474.
    """

    def __init__(
        self,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        joint: bool = True,
        weight_samples: bool = False,
        log_pearson: bool = True,
        **kwargs,
    ):
        # check installation
        if skl_lm is None:
            raise ImportError(
                "This predictor requires an installation of scikit-learn. "
                "Install e.g. via `pip install pyabc[scikit-learn]`")

        default_kwargs = {
            'fit_intercept': True, 'normalize': True,
        }
        default_kwargs.update(kwargs)
        predictor = skl_lm.LinearRegression(**default_kwargs)

        super().__init__(
            predictor=predictor,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            weight_samples=weight_samples,
            log_pearson=log_pearson,
        )

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> None:
        super().fit(x, y, w)
        # log
        if self.joint:
            logger.debug(
                "Linear regression coefficients (n_target, n_feature):\n"
                f"{self.predictor.coef_}")
        else:
            for i_pred, predictor in enumerate(self.single_predictors):
                logger.debug(
                    "Linear regression coefficients (n_target, n_feature):\n"
                    f"for predictor {i_pred}: {predictor.coef_}")


class LassoPredictor(SimplePredictor):
    """Lasso (least absolute shrinkage and selection) model.

    Linear model with l1 regularization.
    """

    def __init__(
        self,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        joint: bool = True,
        log_pearson: bool = True,
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
            'fit_intercept': True,
            'normalize': True,
        }
        all_args.update(kwargs)
        predictor = skl_lm.Lasso(**all_args)

        super().__init__(
            predictor=predictor,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            weight_samples=False,
            log_pearson=log_pearson,
        )


class GPPredictor(SimplePredictor):
    """Gaussian process model.

    Similar to [#borowska2021]_.

    .. [#borowska2021]
        Borowska, Agnieszka, Diana Giurghita, and Dirk Husmeier.
        "Gaussian process enhanced semi-automatic approximate Bayesian
        computation: parameter inference in a stochastic differential equation
        system for chemotaxis."
        Journal of Computational Physics 429 (2021): 109999.
    """

    def __init__(
        self,
        kernel: Union[Callable, skl_gp.kernels.Kernel] = None,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        joint: bool = True,
        log_pearson: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        kernel:
            Covariance kernel. Can be either a kernel, o
        """
        # check installation
        if skl_gp is None:
            raise ImportError(
                "This predictor requires an installation of scikit-learn. "
                "Install e.g. via `pip install pyabc[scikit-learn]`")

        # default kernel
        if kernel is None:
            kernel = GPKernelHandle()
        self.kernel: Union[Callable, skl_gp.kernels.Kernel] = kernel

        self.kwargs = kwargs

        super().__init__(
            predictor=None,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            weight_samples=False,
            log_pearson=log_pearson,
        )

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> None:
        # need to recreate the model

        # set indices to keep
        self.set_use_ixs(x=x, log=False)

        # kernel
        kernel = self.kernel
        if not isinstance(kernel, skl_gp.kernels.Kernel):
            n_in = sum(self.use_ixs)
            kernel = kernel(n_in)

        self.predictor = skl_gp.GaussianProcessRegressor(
            kernel=kernel, **self.kwargs)

        super().fit(x=x, y=y, w=w)


class GPKernelHandle:
    """Convenience class for Gaussian process kernel construction.

    Allows to create kernels depending on problem dimensions.
    """

    # kernels supporting features specific length scales
    ARD_KERNELS = ["RBF", "Matern"]

    def __init__(
        self,
        kernels: List[str] = None,
        kernel_kwargs: List[dict] = None,
        ard: bool = True,
    ):
        """
        Parameters
        ----------
        kernels:
            Names of `sklearn.kernel` covariance kernels. Defaults to a
            radial basis function (a.k.a. squared exponential) kernel "RBF"
            and a "WhiteKernel" to explain noise in the data. The resulting
            kernel is the sum of all kernels.
        kernel_kwargs:
            Optional arguments passed to the kernel constructors.
        ard:
            Automatic relevance determination by assigning a separate length
            scale per input variable. Only supported by some kernels,
            currently "RBF" and "Matern".
            If set to True, the capable kernels are automatically informed.
            It the underlying scitki-learn toolbox extends support, this
            list needs to be updated.
        """
        if kernels is None:
            kernels = ["RBF", "WhiteKernel"]
        self.kernels: List[str] = kernels

        if kernel_kwargs is None:
            kernel_kwargs = [{} for _ in self.kernels]
        self.kernel_kwargs = kernel_kwargs

        self.ard: bool = ard

    def __call__(self, n_in: int) -> "skl_gp.kernels.Kernel":
        """
        Parameters
        ----------
        n_in: Input (feature) dimension.

        Returns
        -------
        kernel: Kernel created from inputs.
        """
        kernels = [
            getattr(skl_gp.kernels, kernel)(
                length_scale=np.ones(n_in), **kernel_kwargs)
            if self.ard and kernel in GPKernelHandle.ARD_KERNELS
            else getattr(skl_gp.kernels, kernel)(**kernel_kwargs)
            for kernel, kernel_kwargs in zip(self.kernels, self.kernel_kwargs)
        ]
        return sum(kernels)


class MLPPredictor(SimplePredictor):
    """Multi-layer perceptron regressor predictor.

    See e.g. [#jiang2017]_.

    .. [#jiang2017]
        Jiang, Bai, et al.
        "Learning summary statistic for approximate Bayesian computation via
        deep neural network."
        Statistica Sinica (2017): 1595-1618.
    """

    def __init__(
        self,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        joint: bool = True,
        hidden_layer_sizes: Union[Tuple[int, ...], Callable] = None,
        log_pearson: bool = True,
        **kwargs,
    ):
        """Additional keyword arguments are passed on to the model.

        Parameters
        ----------
        hidden_layer_sizes:
            Network hidden layer sizes. Can be either a tuple of ints, or
            a callable taking input dimension, output dimension, and number
            of samples and returning a tuple of ints.
            The :class:`HiddenLayerSize` provides some useful defaults.
        """
        # check installation
        if skl_nn is None:
            raise ImportError(
                "This predictor requires an installation of scikit-learn. "
                "Install e.g. via `pip install pyabc[scikit-learn]`")

        if hidden_layer_sizes is None:
            hidden_layer_sizes = HiddenLayerHandle()
        self.hidden_layer_sizes = hidden_layer_sizes

        self.kwargs = {
            'solver': 'lbfgs',
            'max_iter': 10000,
            'early_stopping': True,
        }
        self.kwargs.update(kwargs)

        super().__init__(
            predictor=None,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            weight_samples=False,
            log_pearson=log_pearson,
        )

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> None:
        # need to recreate the model

        # set indices to keep
        self.set_use_ixs(x=x, log=False)

        # hidden layer sizes
        hidden_layer_sizes = self.hidden_layer_sizes
        if callable(hidden_layer_sizes):
            n_in, n_out, n_sample = sum(self.use_ixs), y.shape[1], x.shape[0]
            hidden_layer_sizes = hidden_layer_sizes(n_in, n_out, n_sample)

        self.predictor = skl_nn.MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, **self.kwargs)

        super().fit(x=x, y=y, w=w)


class HiddenLayerHandle:
    """Convenience class for various layer size strategies.

    Allows to define sizes depending on problem dimensions.
    """

    HEURISTIC = "heuristic"
    MEAN = "mean"
    MAX = "max"
    METHODS = [HEURISTIC, MEAN, MAX]

    def __init__(
        self,
        method: Union[str, List[str]] = MEAN,
        n_layer: int = 1,
        max_size: int = np.inf,
        alpha: float = 1.,
    ):
        """
        Parameters
        ----------
        method:
            Method to use. Can be any of:

            * "heuristic" bases the number of neurons on the number of samples
              to avoid overfitting. See
              https://stats.stackexchange.com/questions/181.
            * "mean" takes the mean of input and output dimension.
            * "max" takes the maximum of input and output dimension.

            Additionally, a list of methods can be passed, in which case
            the minimum over all is used.
        n_layer:
            Number of layers.
        max_size:
            Maximum layer size. Applied to all strategies.
        alpha:
            Factor used in "heuristic". The higher, the fewer neurons.
            Recommended is a value in the range 2-10.
        """
        if isinstance(method, str):
            method = [method]
        for m in method:
            if m not in HiddenLayerHandle.METHODS:
                raise ValueError(
                    f"Method {m} must be in {HiddenLayerHandle.METHODS}")
        self.method = method
        self.n_layer = n_layer
        self.max_size = max_size
        self.alpha = alpha

    def __call__(
        self, n_in: int, n_out: int, n_sample: int,
    ) -> Tuple[int, ...]:
        """
        Parameters
        ----------
        n_in: Input (feature) dimension.
        n_out: Output (target) dimension.
        n_sample: Number of samples.

        Returns
        -------
        hidden_layer_sizes: Tuple of hidden layer sizes.
        """
        neurons_arr = []
        for method in self.method:
            if method == HiddenLayerHandle.HEURISTIC:
                # number of neurons
                neurons = n_sample / (self.alpha * (n_in + n_out))
                # divide by number of layers
                neurons /= self.n_layer
            elif method == HiddenLayerHandle.MEAN:
                neurons = 0.5 * (n_in + n_out)
            elif method == HiddenLayerHandle.MAX:
                neurons = max(n_in, n_out)
            else:
                raise ValueError(f"Did not recognize method {self.method}.")
            neurons_arr.append(neurons)

        # take minimum over proposed values
        neurons_per_layer = min(neurons_arr)

        # cap
        neurons_per_layer = min(neurons_per_layer, self.max_size)

        # only >=2 dim makes sense, round
        neurons_per_layer = int(max(2., neurons_per_layer))

        layer_sizes = tuple(neurons_per_layer for _ in range(self.n_layer))
        logger.info(f"Layer sizes: {layer_sizes}")

        return layer_sizes


class ModelSelectionPredictor(Predictor):
    """Model selection over a set of predictors.

    Picks the model with minimum k-fold cross valdation score and retrains on
    full data set.
    """

    CROSS_VALIDATION = "cross_validation"
    TRAIN_TEST_SPLIT = "train_test_split"
    SPLIT_METHODS = [CROSS_VALIDATION, TRAIN_TEST_SPLIT]

    def __init__(
        self,
        predictors: List[Predictor],
        split_method: str = TRAIN_TEST_SPLIT,
        n_splits: int = 5,
        test_size: float = 0.2,
        f_score: Callable = None,
    ):
        """
        Parameters
        ----------
        predictors:
            Set of predictors over which to perform model selection.
        split_method:
            Method how to split the data set into training and test data,
            can be "cross_validation" for a full `n_splits` fold cross
            validation, or "train_test_split" for a single separation of a
            test set of size `test_size`.
        n_splits:
            Number of splits to use in k-fold cross validation.
        test_size:
            Fraction of samples to randomly pick as test set, when using a
            single training and test set.
        f_score:
            Score function to assess prediction quality. Defaults to
            root mean square error normalized by target standard variation.
            Takes arguments y1, y2, std for prediction, ground truth, and
            standard variation, and returns the score as a float.
        """
        super().__init__()
        self.predictors: List[Predictor] = predictors

        if split_method not in ModelSelectionPredictor.SPLIT_METHODS:
            raise ValueError(
                f"Split method {split_method} must be in "
                f"{ModelSelectionPredictor.SPLIT_METHODS}",
            )
        self.split_method: str = split_method

        self.n_splits: int = n_splits
        self.test_size: float = test_size

        if f_score is None:
            self.f_score = root_mean_square_error

        # holds the chosen predictor model
        self.chosen_one: Union[Predictor, None] = None

    def fit(self, x: np.ndarray, y: np.ndarray, w: np.ndarray = None) -> None:
        # output normalization
        std_y = np.std(y, axis=0)

        n_sample = x.shape[0]
        if self.split_method == ModelSelectionPredictor.CROSS_VALIDATION:
            # k-fold cross validation
            kfold = skl_ms.KFold(n_splits=self.n_splits, shuffle=True)
            splits = kfold.split(np.arange(n_sample))
            n_splits = self.n_splits
        else:
            # a single training and test set
            splits = skl_ms.train_test_split(
                np.arange(n_sample), test_size=self.test_size)
            # as iterable
            splits = [splits]
            n_splits = 1

        # scores on training and test set
        scores_test = np.empty((len(self.predictors), n_splits))
        #  for debugging
        scores_train = np.empty((len(self.predictors), n_splits))

        # iterate over training and test sets
        for i_split, (train_ixs, test_ixs) in enumerate(splits):
            # training and test sets
            x_train, y_train = x[train_ixs], y[train_ixs]
            w_train = w
            if w is not None:
                w_train = w[train_ixs]
            x_test, y_test = x[test_ixs], y[test_ixs]

            # iterate over predictors
            for i_pred, predictor in enumerate(self.predictors):
                # fit predictor on training set
                predictor.fit(x=x_train, y=y_train, w=w_train)

                # test score of z-score normalized variables
                y_test_pred = predictor.predict(x=x_test)
                scores_test[i_pred, i_split] = self.f_score(
                    y1=y_test_pred, y2=y_test, sigma=std_y,
                )

                # for debugging, log training scores
                y_train_pred = predictor.predict(x=x_train)
                scores_train[i_pred, i_split] = self.f_score(
                    y1=y_train_pred, y2=y_train, sigma=std_y,
                )

        # the score of a predictor is the mean over all folds
        scores_test = np.mean(scores_test, axis=1)
        scores_train = np.mean(scores_train, axis=1)

        # logging
        for predictor, score_train, score_test in zip(
                self.predictors, scores_train, scores_test):
            logger.info(
                f"Test score {score_test:.3e} (train {score_train:.3e}) for "
                f"{predictor}")

        # best predictor has minimum score
        self.chosen_one = self.predictors[np.argmin(scores_test)]

        # retrain chosen model on full data set
        self.chosen_one.fit(x=x, y=y, w=w)

    def predict(self, x: np.ndarray, normalize: bool = False) -> np.ndarray:
        # predict from the chosen model
        return self.chosen_one.predict(x=x, normalize=normalize)


def root_mean_square_error(
    y1: np.ndarray,
    y2: np.ndarray,
    sigma: Union[np.ndarray, float],
) -> float:
    """Root mean square error of `y1 - y2 / sigma`.

    Parameters
    ----------
    y1: Model simulations, shape (n_sample, n_par).
    y2: Ground truth values, shape (n_sample, n_par).
    sigma: Normalizations, shape (n_sample,) or (1,).

    Returns
    -------
    val: The normalized root mean square error value.
    """
    return np.sqrt(
        np.sum(
            ((y1 - y2) / sigma)**2,
        ) / y1.size,
    )


def root_mean_square_relative_error(
    y1: np.ndarray,
    y2: np.ndarray,
) -> float:
    """Root mean square relative error of `(y1 - y2) / y2`.

    Note that this may behave badly for ground truth parameters close to 0.

    Parameters
    ----------
    y1: Model simulations.
    y2: Ground truth values.

    Returns
    -------
    val: The normalized root mean square relative error value.
    """
    return np.sqrt(
        np.sum(
            ((y1 - y2) / y2)**2,
        ) / y1.size,
    )
