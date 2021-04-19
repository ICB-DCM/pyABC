"""Summary statistics learning."""

import numpy as np
from typing import Callable, List, Union

try:
    import sklearn.linear_model as skl_lm
    import sklearn.gaussian_process as skl_gp
except ImportError as e:
    skl_lm = skl_gp = None

from ..population import Sample
from .util import io_dict2arr, read_sample
from .base import Sumstat, IdentitySumstat
from ..predictor import (
    Predictor,
    SimplePredictor,
    LinearPredictor,
    LassoPredictor,
    GPPredictor,
    MLPPredictor,
)


class PredictorSumstat(Sumstat):
    """
    Summary statistics based on a model predicting parameters from data,
    `y -> theta`.
    For some predictor models, there exist dedicated subclasses.

    The predictor should define:

    - `fit(X, y)` to fit the model on a sample of data X and outputs y,
      where X has shape (n_sample, n_feature), and
      y has shape (n_sample, n_out), with n_out either the parameter dimension
      or 1, depending on `joint`.
      Further, `fit(X, y, weights)` gets as a third argument the sample weights
      if `weight_samples` is set. Not all predictors support this.
    - `predict(X)` to predict outputs of shape (n_out,), where X has shape
      (n_sample, n_feature).
    """

    def __init__(
        self,
        predictor: Union[Predictor, Callable],
        n_fit: int = 1,
        all_particles: bool = True,
        normalize_labels: bool = True,
        pre: Sumstat = None,
    ):
        """
        Parameters
        ----------
        predictor:
            The predictor mapping data (inputs) to parameters (targets). See
            :class:`Predictor` for the functionality contract.
        n_fit:
            Number of generations in which the model is updated.
            1 means only at the beginning in `initialize`, for > 1 also in
            subsequent generations in `update`. Can be inf.
        all_particles:
            Whether to base the predictors on all samples, or only accepted
            ones. Basing it on all samples reflects the sample process,
            while only considering accepted particles (and additionally
            weighting them) reflects the posterior approximation.
        normalize_labels:
            Whether the outputs in `__call__` are normalized according to
            potentially applied internal normalization of the predictor.
            This allows to level the influence of labels.
        pre:
            Previously applied summary statistics, enables chaining. Should
            usually not be adaptive.
        """
        if pre is None:
            pre = IdentitySumstat()
        super().__init__(pre=pre)

        if not isinstance(predictor, Predictor):
            predictor = SimplePredictor(predictor=predictor)
        self.predictor = predictor

        self.n_fit: int = n_fit
        self.all_particles: bool = all_particles
        self.normalize_labels: bool = normalize_labels

        # parameter keys (for correct order)
        self.par_keys: Union[List[str], None] = None

        # fit counter
        self.i_fit: int = 0

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict = None,
    ) -> None:
        super().initialize(t=t, get_sample=get_sample, x_0=x_0)

        # call cached function
        sample = get_sample()

        # fix parameter key order
        self.par_keys: List[str] = \
            list(sample.accepted_particles[0].parameter.keys())

        # check whether refitting is desired
        if self.i_fit >= self.n_fit:
            return
        else:
            self.i_fit += 1

        # extract information from sample
        sumstats, parameters, weights = read_sample(
            sample=sample, sumstat=self.pre, all_particles=self.all_particles,
            par_keys=self.par_keys,
        )

        # fit model to sample
        self.predictor.fit(x=sumstats, y=parameters, w=weights)

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample],
    ) -> bool:
        updated = super().update(t=t, get_sample=get_sample)

        # check whether refitting is desired
        if self.i_fit >= self.n_fit:
            return updated
        else:
            self.i_fit += 1

        # call cached function
        sample = get_sample()

        # extract information from sample
        sumstats, parameters, weights = read_sample(
            sample=sample, sumstat=self.pre, all_particles=self.all_particles,
            par_keys=self.par_keys,
        )

        # fit model to sample
        self.predictor.fit(x=sumstats, y=parameters, w=weights)

        return True

    def configure_sampler(self, sampler) -> None:
        if self.n_fit > 1 and self.all_particles:
            # record rejected particles as a more representative of the
            #  sampling process
            sampler.sample_factory.record_rejected()

    def requires_calibration(self) -> bool:
        return self.n_fit > 0 or self.pre.requires_calibration()

    def is_adaptive(self) -> bool:
        return self.n_fit > 1 or self.pre.is_adaptive()

    @io_dict2arr
    def __call__(self, data: Union[dict, np.ndarray]):
        data = self.pre(data)

        # summary statistic is the (normalized) predictor value
        sumstat = self.predictor.predict(
            data, normalize=self.normalize_labels).flatten()

        if sumstat.size != len(self.par_keys):
            raise AssertionError("Predictor should return #parameters values")

        return sumstat

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} pre={self.pre}, " \
               f"predictor={self.predictor}>"

    def get_ids(self) -> List[str]:
        # label by parameter keys
        return [f"s_{par_key}" for par_key in self.par_keys]


# Convenience wrappers around the respective predictors


class LinearPredictorSumstat(PredictorSumstat):
    """Use a linear model for `y -> theta`.

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
        n_fit: int = 1,
        joint: bool = True,
        weight_samples: bool = False,
        pre: Sumstat = None,
    ):
        """
        Parameters
        -----------
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
        predictor = LinearPredictor(
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            weight_samples=weight_samples,
        )

        super().__init__(
            predictor=predictor,
            n_fit=n_fit,
            pre=pre,
        )


class LassoPredictorSumstat(PredictorSumstat):
    """Lasso (least absolute shrinkage and selection) model for `y -> theta`.
    """
    def __init__(
        self,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        n_fit: int = 1,
        joint: bool = True,
        pre: Sumstat = None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        normalize_features:
            Whether to apply z-score normalization to the input data.
        normalize_labels:
            Whether to apply z-score normalization to the parameters.
        joint:
            Whether the predictor learns one model for all targets, or
            separate models per target.
        """
        predictor = LassoPredictor(
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            **kwargs,
        )

        super().__init__(
            predictor=predictor,
            n_fit=n_fit,
            pre=pre,
        )


class GPPredictorSumstat(PredictorSumstat):
    """Use a Gaussian Process model for `y -> theta`.

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
        kernel=None,
        normalize_features: bool = True,
        normalize_labels: bool = True,
        n_fit: int = 1,
        joint: bool = True,
        pre: Sumstat = None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        kernel:
            Gaussian process covariance function.
        normalize_features:
            Whether to apply z-score normalization to the input data.
        normalize_labels:
            Whether to apply z-score normalization to the parameters.
        joint:
            Whether the predictor learns one model for all targets, or
            separate models per target.
        """
        predictor = GPPredictor(
            kernel=kernel,
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            **kwargs,
        )

        super().__init__(
            predictor=predictor,
            n_fit=n_fit,
            pre=pre,
        )


class MLPPredictorSumstat(PredictorSumstat):
    """Use a multi-layer perceptron model for `y -> theta`.

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
        n_fit: int = 1,
        joint: bool = True,
        pre: Sumstat = None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        kernel:
            Gaussian process covariance function.
        normalize_features:
            Whether to apply z-score normalization to the input data.
        normalize_labels:
            Whether to apply z-score normalization to the parameters.
        joint:
            Whether the predictor learns one model for all targets, or
            separate models per target.
        """
        predictor = MLPPredictor(
            normalize_features=normalize_features,
            normalize_labels=normalize_labels,
            joint=joint,
            **kwargs,
        )

        super().__init__(
            predictor=predictor,
            n_fit=n_fit,
            pre=pre,
        )
