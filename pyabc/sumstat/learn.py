"""Summary statistics learning."""

import logging
from typing import Callable, Collection, List, Union

import numpy as np

try:
    import sklearn.gaussian_process as skl_gp
    import sklearn.linear_model as skl_lm
except ImportError:
    skl_lm = skl_gp = None

from ..population import Sample
from ..predictor import Predictor, SimplePredictor
from ..util import (
    EventIxs,
    ParTrafo,
    ParTrafoBase,
    dict2arrlabels,
    io_dict2arr,
    read_sample,
)
from .base import IdentitySumstat, Sumstat
from .subset import IdSubsetter, Subsetter

logger = logging.getLogger("ABC.Sumstat")


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
        fit_ixs: Union[EventIxs, Collection[int], int] = None,
        all_particles: bool = True,
        normalize_labels: bool = True,
        fitted: bool = False,
        subsetter: Subsetter = None,
        pre: Sumstat = None,
        pre_before_fit: bool = False,
        par_trafo: ParTrafoBase = None,
    ):
        """
        Parameters
        ----------
        predictor:
            The predictor mapping data (inputs) to parameters (targets). See
            :class:`Predictor` for the functionality contract.
        fit_ixs:
            Generation indices when to (re)fit the model, e.g. `{9, 15}`.
            See :class:`pyabc.EventIxs` for possible values.
            In generations before the first fit, the output of `pre` is
            returned as-is.
        all_particles:
            Whether to base the predictors on all samples, or only accepted
            ones. Basing it on all samples reflects the sampling process,
            while only considering accepted particles (and additionally
            weighting them) reflects the posterior approximation.
        normalize_labels:
            Whether the outputs in `__call__` are normalized according to
            potentially applied internal normalization of the predictor.
            This allows to level the influence of labels.
        fitted:
            Set to True if the predictor model passed has aready been fitted
            externally.
            If False, the `__call__` function will return the
            output of `pre` until the first time index in `fit_ixs`.
        subsetter:
            Sample subset/cluster selection method. Defaults to just taking all
            samples. In the presence of e.g. multi-modalities it may make sense
            to reduce.
        pre:
            Previously applied summary statistics, enables chaining. Should
            usually not be adaptive.
        pre_before_fit:
            Apply previous summary statistics also before any fit is performed,
            or just return the input then and only apply pre when
            regression-based summary statistics are calculated.
        par_trafo:
            Parameter transformations to use as targets. Defaults to identity.
        """
        if pre is None:
            pre = IdentitySumstat()
        super().__init__(pre=pre)

        if not isinstance(predictor, Predictor):
            predictor = SimplePredictor(predictor=predictor)
        self.predictor = predictor

        if fit_ixs is None:
            fit_ixs = {9, 15}
        self.fit_ixs: EventIxs = EventIxs.to_instance(fit_ixs)
        logger.debug(f"Fit model ixs: {self.fit_ixs}")

        self.all_particles: bool = all_particles
        self.normalize_labels: bool = normalize_labels

        # indicate whether the model has ever been fitted
        self.fitted: bool = fitted

        if subsetter is None:
            subsetter = IdSubsetter()
        self.subsetter: Subsetter = subsetter

        self.pre_before_fit: bool = pre_before_fit

        if par_trafo is None:
            par_trafo = ParTrafo()
        self.par_trafo: ParTrafoBase = par_trafo

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ) -> None:
        super().initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )

        # call cached function
        sample = get_sample()

        # fix parameter key order
        self.par_trafo.initialize(
            keys=list(sample.accepted_particles[0].parameter.keys()),
        )

        # check whether to skip fitting
        if not self.fit_ixs.act(t=t, total_sims=total_sims):
            return

        # extract information from sample
        sumstats, parameters, weights = read_sample(
            sample=sample,
            sumstat=self.pre,
            all_particles=self.all_particles,
            par_trafo=self.par_trafo,
        )

        # subset sample
        sumstats, parameters, weights = self.subsetter.select(
            x=sumstats,
            y=parameters,
            w=weights,
        )

        # fit model to sample
        self.predictor.fit(x=sumstats, y=parameters, w=weights)
        self.fitted = True

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        total_sims: int,
    ) -> bool:
        updated = super().update(
            t=t,
            get_sample=get_sample,
            total_sims=total_sims,
        )

        # check whether to skip fitting
        if not self.fit_ixs.act(t=t, total_sims=total_sims):
            return updated

        # call cached function
        sample = get_sample()

        # extract information from sample
        sumstats, parameters, weights = read_sample(
            sample=sample,
            sumstat=self.pre,
            all_particles=self.all_particles,
            par_trafo=self.par_trafo,
        )

        # subset sample
        sumstats, parameters, weights = self.subsetter.select(
            x=sumstats,
            y=parameters,
            w=weights,
        )

        # fit model to sample
        self.predictor.fit(x=sumstats, y=parameters, w=weights)
        self.fitted = True

        return True

    def configure_sampler(self, sampler) -> None:
        if self.all_particles and self.fit_ixs.probably_has_late_events():
            # record rejected particles as a more representative of the
            #  sampling process
            sampler.sample_factory.record_rejected()

    def requires_calibration(self) -> bool:
        return (
            self.fit_ixs.requires_calibration()
            or self.pre.requires_calibration()
        )

    def is_adaptive(self) -> bool:
        return (
            self.fit_ixs.probably_has_late_events() or self.pre.is_adaptive()
        )

    @io_dict2arr
    def __call__(self, data: Union[dict, np.ndarray]):
        # check whether to return data directly
        if not self.fitted and not self.pre_before_fit:
            return data

        data = self.pre(data)

        # check whether to return pre-sumstat data directly
        if not self.fitted:
            return data

        # summary statistic is the (normalized) predictor value
        sumstat = self.predictor.predict(
            data, normalize=self.normalize_labels
        ).flatten()

        if sumstat.size != len(self.par_trafo):
            raise AssertionError("Predictor should return #parameters values")

        return sumstat

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} pre={self.pre}, "
            f"predictor={self.predictor}>"
        )

    def get_ids(self) -> List[str]:
        # label by parameter keys
        if self.fitted:
            return [f"s_{key}" for key in self.par_trafo.get_ids()]
        if not self.pre_before_fit:
            return dict2arrlabels(self.x_0, keys=self.x_keys)
        return self.pre.get_ids()
