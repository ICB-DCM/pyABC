"""Distance functions."""
from abc import ABC
import numpy as np
from scipy import linalg as la
from numbers import Number
from typing import Callable, Collection, Dict, List, Union
import logging

from ..storage import save_dict_to_json
from ..population import Sample
from ..predictor import Predictor
from ..sumstat import (
    Sumstat, IdentitySumstat, Subsetter, IdSubsetter,
)
from ..util import (
    dict2arr, read_sample, ParTrafoBase, ParTrafo, EventIxs, log_samples,
)

from .scale import mad, span
from .base import Distance, to_distance
from .util import bound_weights, log_weights, fd_nabla1_multi_delta


logger = logging.getLogger("ABC.Distance")


class PNormDistance(Distance):
    """Weighted p-norm distance.

    Distance between summary statistics calculated according to

    .. math::

        d(x, y) = \
        \\left [\\sum_{i} \\left| w_i ( x_i-y_i ) \\right|^{p} \\right ]^{1/p}

    E.g.
    * p=1 for a Euclidean or L1 metric,
    * p=2 for a Manhattan or L2 metric,
    * p=np.inf for a Chebyshev, maximum or inf metric.

    Parameters
    ----------
    p:
        p for p-norm. p >= 1, p = np.inf implies max norm.
    fixed_weights:
        Weights.
        Dictionary indexed by time points, or only one entry if
        the weights should not change over time.
        Each entry contains a dictionary of numeric weights, indexed by summary
        statistics labels, or the corresponding array representation.
        If None is passed, a weight of 1 is considered for every summary
        statistic.
        If no entry is available for a given time point, the maximum available
        time point is selected.
    sumstat:
        Summary statistics transformation to apply to the model output.
    """

    def __init__(
        self,
        p: float = 1,
        fixed_weights: Union[Dict[str, float],
                             Dict[int, Dict[str, float]]] = None,
        sumstat: Sumstat = None,
    ):
        super().__init__()

        if sumstat is None:
            sumstat = IdentitySumstat()
        self.sumstat: Sumstat = sumstat

        if p < 1:
            raise ValueError("It must be p >= 1")
        self.p: float = p

        self._arg_fixed_weights = fixed_weights
        self.fixed_weights: Union[Dict[int, np.ndarray], None] = None

        # to cache the observed data and summary statistics
        self.x_0: Union[dict, None] = None
        self.s_0: Union[np.ndarray, None] = None

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ) -> None:
        # update summary statistics
        self.sumstat.initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )

        # observed data
        self.x_0 = x_0
        self.s_0 = self.sumstat(self.x_0)

        # initialize weights
        self.fixed_weights = PNormDistance.format_dict(
            vals=self._arg_fixed_weights,
            t=t,
            s_ids=self.sumstat.get_ids(),
        )

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        total_sims: int,
    ) -> bool:
        # update summary statistics
        updated = self.sumstat.update(
            t=t,
            get_sample=get_sample,
            total_sims=total_sims,
        )
        if updated:
            self.s_0 = self.sumstat(self.x_0)
        return updated

    def configure_sampler(self, sampler) -> None:
        self.sumstat.configure_sampler(sampler=sampler)

    def get_weights(self, t: int) -> np.ndarray:
        """Compute weights for time `t`.

        Generates weights from the multiple possible contributing factors.
        Overwrite in subclasses if there are additional weights.

        Parameters
        ----------
        t: Current time point.

        Returns
        -------
        The combined weights.
        """
        fixed_weights: np.ndarray = \
            PNormDistance.for_t_or_latest(self.fixed_weights, t)
        return fixed_weights

    @staticmethod
    def format_dict(
        vals: Union[Dict[str, float], Dict[int, Dict[str, float]]],
        t: int,
        s_ids: List[str],
    ) -> Dict[int, Union[float, np.ndarray]]:
        """Normalize weight dictionary to the employed format.

        Parameters
        ----------
        vals: Possibly unformatted weight values.
        t: Current time point.
        s_ids: Summary statistic labels for correct conversion to array.

        Returns
        -------
        Dictionary indexed by time points and, with array values.
        """
        if vals is None:
            # use default
            vals = {t: 1.}
            return vals
        elif isinstance(next(iter(vals.values())), Number):
            vals = {t: vals}

        # convert dicts to arrays
        for _t, dct in vals.items():
            vals[_t] = dict2arr(dct, keys=s_ids)

        return vals

    @staticmethod
    def for_t_or_latest(w: Dict[int, np.ndarray], t: int) -> np.ndarray:
        """Extract values from dict for given time point.

        Parameters
        ----------
        w: Weights dictionary.
        t: Time point to extract weights for.

        Returns
        -------
        The The weights at time t, or the maximal key if t is not present.
        """
        # take last time point for which values exist
        if t not in w:
            smaller_ts = [t_ for t_ in w.keys() if t_ <= t]
            if len(smaller_ts) == 0:
                return np.asarray(1.)
            t = max(smaller_ts)
        # extract values for time point
        return np.asarray(w[t])

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None,
    ) -> float:
        # extract weights for given time point
        weights = self.get_weights(t=t)

        # compute summary statistics
        s, s0 = self.sumstat(x), self.sumstat(x_0)

        # assert shapes match
        if s.shape != weights.shape and weights.shape or s.shape != s0.shape:
            raise AssertionError(
                f"Shapes do not match: s={s.shape}, s0={s0.shape}, "
                f"weights={weights.shape}")

        # component-wise distances
        distances = np.abs(weights * (s - s0))

        # maximum or p-norm distance
        if self.p == np.inf:
            return distances.max()
        return (distances**self.p).sum()**(1/self.p)

    def get_config(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "p": self.p,
            "fixed_weights": self.fixed_weights,
            "sumstat": self.sumstat.__str__(),
        }

    def weights2dict(
        self, weights: Dict[int, np.ndarray],
    ) -> Dict[int, Dict[str, float]]:
        """Create labeled weights dictionary.

        Parameters
        ----------
        weights:
            Array formatted weight dictionary.

        Returns
        -------
        weights_dict: Key-value formatted weight dictionary.
        """
        # assumes that the summary statistic labels do not change over time
        return {
            t: {
                key: val
                for key, val in zip(self.sumstat.get_ids(), weights[t])
            }
            for t in weights
        }


class AdaptivePNormDistance(PNormDistance):
    """
    In the p-norm distance, adapt the weights for each generation, based on
    the previous simulations. This class is motivated by [#prangle]_.

    Parameters
    ----------
    p:
        p for p-norm. Required p >= 1, p = np.inf allowed (infinity-norm).
    initial_scale_weights:
        Scale weights to be used in the initial iteration. Dictionary with
        observables as keys and weights as values.
    fixed_weights:
        Fixed multiplicative factors the weights are multiplied with, to
        e.g. account for heterogeneous numbers of data points.
        The discrimination of various weight types makes only sense for
        adaptive distances.
    fit_scale_ixs:
        Generation indices before which to (re)fit the scale weights.
        Inf (default) means in every generation. For other values see
        :class:`pyabc.EventIxs`.
    scale_function:
        (data: list, x_0: float) -> scale: float. Computes the scale (i.e.
        inverse weight s = 1 / w) for a given summary statistic. Here, data
        denotes the list of simulated summary statistics, and x_0 the observed
        summary statistic. Implemented are absolute_median_deviation,
        standard_deviation (default), centered_absolute_median_deviation,
        centered_standard_deviation.
    max_scale_weight_ratio:
        If not None, extreme scale weights will be bounded by the ratio times
        the smallest non-zero absolute scale weight.
        In practice usually not necessary, it is theoretically required to
        ensure convergence if weights are refitted in infinitely many
        iterations.
    scale_log_file:
        A log file to store scale weights for each time point in. Weights are
        currently not stored in the database. The data are saved in json
        format and can be retrieved via `pyabc.storage.load_dict_from_json`.
    all_particles_for_scale:
        Whether to include also rejected particles for scale calculation
        (True) or only accepted ones (False).
    sumstat:
        Summary statistics. Defaults to an identity mapping.


    .. [#prangle] Prangle, Dennis.
        "Adapting the ABC Distance Function".
        Bayesian Analysis, 2017.
        https://doi.org/10.1214/16-BA1002
    """

    def __init__(
        self,
        p: float = 1,
        initial_scale_weights: Dict[str, float] = None,
        fixed_weights: Dict[str, float] = None,
        fit_scale_ixs: Union[EventIxs, Collection[int], int] = np.inf,
        scale_function: Callable = None,
        max_scale_weight_ratio: float = None,
        scale_log_file: str = None,
        all_particles_for_scale: bool = True,
        sumstat: Sumstat = None,
    ):
        # call p-norm constructor
        super().__init__(p=p, fixed_weights=fixed_weights, sumstat=sumstat)

        self.initial_scale_weights: Dict[str, float] = initial_scale_weights

        self.scale_weights: Dict[int, np.ndarray] = {}

        # extract indices when to fit scales from input
        if fit_scale_ixs is None:
            fit_scale_ixs = {np.inf}
        self.fit_scale_ixs: EventIxs = EventIxs.to_instance(fit_scale_ixs)
        logger.debug(f"Fit scale ixs: {self.fit_scale_ixs}")

        if scale_function is None:
            scale_function = mad
        self.scale_function: Callable = scale_function

        self.max_scale_weight_ratio: float = max_scale_weight_ratio
        self.scale_log_file: str = scale_log_file
        self.all_particles_for_scale: bool = all_particles_for_scale

    def configure_sampler(self, sampler) -> None:
        """
        Make the sampler return also rejected particles,
        because these are needed to get a better estimate of the summary
        statistic variability, avoiding a bias to accepted ones only.

        Parameters
        ----------
        sampler: Sampler
            The sampler employed.
        """
        super().configure_sampler(sampler=sampler)
        if self.all_particles_for_scale and \
                self.fit_scale_ixs.probably_has_late_events():
            sampler.sample_factory.record_rejected()

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ) -> None:
        # esp. initialize sumstats
        super().initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )

        # are initial weights pre-defined
        if self.initial_scale_weights is not None:
            self.scale_weights[t] = dict2arr(
                self.initial_scale_weights, keys=self.sumstat.get_ids())
            return

        if not self.fit_scale_ixs.act(t=t, total_sims=total_sims):
            raise ValueError(
                f"Initial scale weights (t={t}) must be fitted or provided.")

        # execute cached function
        sample = get_sample()

        # update weights from samples
        self.fit_scales(t=t, sample=sample)

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        total_sims: int,
    ) -> bool:
        # esp. updates summary statistics
        updated = super().update(
            t=t,
            get_sample=get_sample,
            total_sims=total_sims,
        )

        # check whether weight fitting is desired
        if not self.fit_scale_ixs.act(t=t, total_sims=total_sims):
            if updated:
                logger.warning(
                    f"t={t}: Updated sumstat but not scale weights.")
            return updated

        # execute cached function
        sample = get_sample()

        self.fit_scales(t=t, sample=sample)

        return True

    def fit_scales(
        self,
        t: int,
        sample: Sample,
    ) -> None:
        """Here the real weight update happens."""
        # create (n_sample, n_feature) matrix of all summary statistics
        if self.all_particles_for_scale:
            particles = sample.all_particles
        else:
            particles = sample.accepted_particles
        ss = np.array(
            [self.sumstat(p.sum_stat).flatten() for p in particles],
        )

        # observed summary statistics
        s0 = self.s_0

        # calculate scales
        scales = self.scale_function(
            samples=ss, s0=s0, s_ids=self.sumstat.get_ids(), t=t)

        # weights are the inverse scales
        # a scale close to zero happens e.g. if all simulations are identical
        # in any case, it should be safe to ignore this statistic
        weights = np.zeros_like(scales)
        weights[~np.isclose(scales, 0)] = 1 / scales[~np.isclose(scales, 0)]

        # bound weights
        weights = bound_weights(
            weights, max_weight_ratio=self.max_scale_weight_ratio)

        # update weights attribute
        self.scale_weights[t] = weights

        # logging
        log_weights(
            t=t, weights=self.scale_weights, keys=self.sumstat.get_ids(),
            label="Scale", log_file=self.scale_log_file)

    def get_weights(self, t: int) -> np.ndarray:
        scale_weights: np.ndarray = \
            PNormDistance.for_t_or_latest(self.scale_weights, t)
        return super().get_weights(t=t) * scale_weights

    def requires_calibration(self) -> bool:
        if self.initial_scale_weights is None:
            return True
        return self.sumstat.requires_calibration()

    def is_adaptive(self) -> bool:
        if self.fit_scale_ixs.probably_has_late_events():
            return True
        return self.sumstat.is_adaptive()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "fit_scale_ixs": self.fit_scale_ixs.__repr__(),
            "scale_function": self.scale_function.__name__,
            "max_scale_weight_ratio": self.max_scale_weight_ratio,
            "all_particles_for_scale": self.all_particles_for_scale,
        })
        return config


class InfoWeightedPNormDistance(AdaptivePNormDistance):
    """Weight summary statistics by sensitivity of a predictor `y -> theta`."""

    WEIGHTS = "weights"
    STD = "std"
    MAD = "mad"
    NONE = "none"
    FEATURE_NORMALIZATIONS = [WEIGHTS, STD, MAD, NONE]

    def __init__(
        self,
        predictor: Predictor,
        p: float = 1,
        initial_scale_weights: Dict[str, float] = None,
        initial_info_weights: Dict[str, float] = None,
        fixed_weights: Dict[str, float] = None,
        fit_scale_ixs: Union[EventIxs, Collection, int] = np.inf,
        fit_info_ixs: Union[EventIxs, Collection, int] = None,
        normalize_by_par: bool = True,
        scale_function: Callable = None,
        max_scale_weight_ratio: float = None,
        max_info_weight_ratio: float = None,
        scale_log_file: str = None,
        info_log_file: str = None,
        info_sample_log_file: str = None,
        sumstat: Sumstat = None,
        fd_deltas: Union[List[float], float] = None,
        subsetter: Subsetter = None,
        all_particles_for_scale: bool = True,
        all_particles_for_prediction: bool = True,
        feature_normalization: str = WEIGHTS,
        par_trafo: ParTrafoBase = None,
    ):
        """
        Parameters
        ----------
        predictor:
            Predictor model used to quantify the information in data on
            parameters.
        initial_info_weights:
            Initial information weights. Can be passed to avoid
            (re)-calibration.
        fit_info_ixs:
            Generations when to fit the information weights, similar to
            `fit_scale_ixs`.
            Defaults to {9, 15}, which may not always be the smartest choice.
            In particular consider making it dependent on the total number of
            simulations.
        normalize_by_par:
            Whether to normalize total sensitivities of each parameter to 1.
        max_info_weight_ratio:
            Maximum ratio on information weights, similar to
            `max_scale_weight_ratio`.
        info_log_file:
            Log file for the information weights.
        info_sample_log_file:
            Log file for samples used to train the regression model underlying
            the information weights, in npy format.
            Should be only a base file name, will be automatically postfixed
            by "{t}_{var}.npy", with var in samples, parameters, weights.
        fd_deltas:
            Finite difference step sizes. Can be a float, or a List of floats,
            in which case component-wise step size selection is employed.
        subsetter:
            Sample subset/cluster selection method. Defaults to just taking all
            samples. In the presence of e.g. multi-modalities it may make sense
            to reduce.
        all_particles_for_scale:
            Whether to use all particles for scale calculation (True) or only
            accepted ones (False).
        all_particles_for_prediction:
            Whether to include rejected particles for fitting predictor models.
            The same arguments apply as for `PredictorSumstat.all_particles`,
            i.e. not using all may yield a better local approximation.
        feature_normalization:
            What normalization to apply to the parameters before predictor
            model fitting. Can be any of "std" (standard deviation), "mad"
            (median absolute deviation), "weights" (use the inverse scale
            weights), or "none" (no normalization). It is recommended to
            match this with the `scale_function`, e.g. std or mad. Allowing
            to specify different normalizations (and not "weights") allows
            to e.g. employ outlier down-weighting in the scale function,
            and just normalize differently here, in order to not counteract
            that.
        par_trafo:
            Parameter transformations to use as targets. Defaults to identity.
        """
        # call p-norm constructor
        super().__init__(
            p=p,
            initial_scale_weights=initial_scale_weights,
            fixed_weights=fixed_weights,
            fit_scale_ixs=fit_scale_ixs,
            scale_function=scale_function,
            max_scale_weight_ratio=max_scale_weight_ratio,
            scale_log_file=scale_log_file,
            all_particles_for_scale=all_particles_for_scale,
            sumstat=sumstat,
        )

        self.predictor = predictor

        self.initial_info_weights: Dict[str, float] = initial_info_weights
        self.info_weights: Dict[int, np.ndarray] = {}

        if fit_info_ixs is None:
            fit_info_ixs = {9, 15}
        self.fit_info_ixs: EventIxs = EventIxs.to_instance(fit_info_ixs)
        logger.debug(f"Fit info ixs: {self.fit_info_ixs}")

        self.normalize_by_par: bool = normalize_by_par
        self.max_info_weight_ratio: float = max_info_weight_ratio
        self.info_log_file: str = info_log_file
        self.info_sample_log_file: str = info_sample_log_file
        self.fd_deltas: Union[List[float], float] = fd_deltas

        if subsetter is None:
            subsetter = IdSubsetter()
        self.subsetter: Subsetter = subsetter

        self.all_particles_for_prediction: bool = all_particles_for_prediction

        if feature_normalization not in \
                InfoWeightedPNormDistance.FEATURE_NORMALIZATIONS:
            raise ValueError(
                f"Feature normalization {feature_normalization} must be in "
                f"{InfoWeightedPNormDistance.FEATURE_NORMALIZATIONS}",
            )
        self.feature_normalization: str = feature_normalization

        if par_trafo is None:
            par_trafo = ParTrafo()
        self.par_trafo: ParTrafoBase = par_trafo

    def configure_sampler(self, sampler) -> None:
        """
        Make the sampler return also rejected particles,
        because these are needed to get a better estimate of the summary
        statistic variability, avoiding a bias to accepted ones only.

        Parameters
        ----------

        sampler: Sampler
            The sampler employed.
        """
        super().configure_sampler(sampler=sampler)
        if self.all_particles_for_prediction and \
                self.fit_info_ixs.probably_has_late_events():
            sampler.sample_factory.record_rejected()

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ) -> None:
        # esp. initialize sumstats, weights
        super().initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )

        # are initial weights pre-defined
        if self.initial_info_weights is not None:
            self.info_weights[t] = dict2arr(
                self.initial_info_weights, keys=self.sumstat.get_ids())
            return

        if not self.fit_info_ixs.act(t=t, total_sims=total_sims):
            return

        # execute cached function
        sample = get_sample()

        # initialize parameter transformations
        self.par_trafo.initialize(
            list(sample.accepted_particles[0].parameter.keys()),
        )

        # update weights from samples
        self.fit_info(t=t, sample=sample)

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        total_sims: int,
    ) -> bool:
        # esp. updates summary statistics, weights
        updated = super().update(
            t=t,
            get_sample=get_sample,
            total_sims=total_sims,
        )

        # check whether refitting is desired
        if not self.fit_info_ixs.act(t=t, total_sims=total_sims):
            return updated

        # execute cached function
        sample = get_sample()

        self.fit_info(t=t, sample=sample)

        return True

    def fit_info(
        self,
        t: int,
        sample: Sample,
    ) -> None:
        """Update information weights from model fits."""
        # create (n_sample, n_feature) matrix of all summary statistics
        sumstats, parameters, weights = read_sample(
            sample=sample, sumstat=self.sumstat,
            all_particles=self.all_particles_for_prediction,
            par_trafo=self.par_trafo,
        )
        # log samples used for training
        log_samples(
            t=t, sumstats=sumstats, parameters=parameters, weights=weights,
            log_file=self.info_sample_log_file)

        s_0 = self.sumstat(self.x_0)

        # normalize features and labels
        ret = InfoWeightedPNormDistance.normalize_sample(
            sumstats=sumstats,
            parameters=parameters,
            weights=weights,
            s_0=s_0,
            t=t,
            subsetter=self.subsetter,
            feature_normalization=self.feature_normalization,
            scale_weights=self.scale_weights,
        )
        x, y, weights, use_ixs, x0 = \
            (ret[key] for key in ("x", "y", "weights", "use_ixs", "x0"))

        # learn predictor model
        self.predictor.fit(x=x, y=y, w=weights)

        # calculate all sensitivities of the predictor at the observed data
        sensis = InfoWeightedPNormDistance.calculate_sensis(
            predictor=self.predictor,
            fd_deltas=self.fd_deltas,
            x0=x0,
            n_x=x.shape[1],
            n_y=y.shape[1],
            par_trafo=self.par_trafo,
            normalize_by_par=self.normalize_by_par,
        )

        # the weight of a sumstat is the sum of the sensitivities over all
        #  parameters
        info_weights_red = np.sum(sensis, axis=1)

        if np.allclose(info_weights_red, 0):
            info_weights_red = 1 * np.ones_like(info_weights_red)
            logger.info("All info weights zeros, thus resetting to ones.")
        else:
            # in order to make each sumstat count a little, avoid zero values
            zero_info = np.isclose(info_weights_red, 0)
            # minimum non-zero entry
            min_info = np.min(info_weights_red[~zero_info])
            info_weights_red[zero_info] = min_info / 10

        # project onto full sumstat vector and normalize by scale
        info_weights = np.zeros_like(s_0, dtype=float)
        use_ixs = np.asarray(use_ixs, dtype=bool)

        info_weights[use_ixs] = info_weights_red

        # bound weights
        info_weights = bound_weights(
            info_weights, max_weight_ratio=self.max_info_weight_ratio)

        # update weights attribute
        self.info_weights[t] = info_weights

        # logging
        log_weights(
            t=t, weights=self.info_weights, keys=self.sumstat.get_ids(),
            label="Info", log_file=self.info_log_file)

    @staticmethod
    def normalize_sample(
        sumstats: np.ndarray,
        parameters: np.ndarray,
        weights: np.ndarray,
        s_0: np.ndarray,
        t: int,
        subsetter: Subsetter,
        feature_normalization: str,
        scale_weights: Dict[int, np.ndarray],
    ) -> Dict:
        """Normalize samples prior to regression model training.

        Parameters
        ----------
        sumstats: Model outputs or summary statistics, shape (n_sample, n_x).
        parameters: Parameter values, shape (n_sample, n_y).
        weights: Importance sampling weights, shape (n_sample,).
        s_0: Observed data, shape (n_x,).
        t: Time point, only needed together with scale_weights.
        subsetter: Subset creator.
        feature_normalization: Method of feature normalization.
        scale_weights:
            Dictionary of scale weights, only used if
            feature_normalization=="weights".

        Returns
        -------
        ret:
            Dictionary with keys x, y, weights, use_ixs, x0.
        """
        # subset sample
        sumstats, parameters, weights = subsetter.select(
            x=sumstats, y=parameters, w=weights,
        )

        # define features
        x = sumstats

        # define feature scaling
        if feature_normalization == InfoWeightedPNormDistance.WEIGHTS:
            if scale_weights is None:
                raise ValueError("Requiested scale weights but None passed")
            scale_weights = scale_weights[t]
            offset_x = np.zeros_like(scale_weights)
            scale_x = np.zeros_like(scale_weights)
            use_ixs = ~np.isclose(scale_weights, 0.)
            scale_x[use_ixs] = 1. / scale_weights[use_ixs]
        elif feature_normalization == InfoWeightedPNormDistance.STD:
            # std
            offset_x = np.nanmean(x, axis=0)
            scale_x = np.nanstd(x, axis=0)
        elif feature_normalization == InfoWeightedPNormDistance.MAD:
            offset_x = np.nanmedian(x, axis=0)
            scale_x = np.nanmedian(np.abs(x - offset_x), axis=0)
        elif feature_normalization == InfoWeightedPNormDistance.NONE:
            offset_x = np.zeros(shape=x.shape[1])
            scale_x = np.ones(shape=x.shape[1])
        else:
            raise ValueError(
                f"Feature normalization {feature_normalization} must be "
                f"in {InfoWeightedPNormDistance.FEATURE_NORMALIZATIONS}",
            )

        # remove trivial features
        use_ixs = ~np.isclose(scale_x, 0.)
        x, offset_x, scale_x = \
            x[:, use_ixs], offset_x[use_ixs], scale_x[use_ixs]

        # normalize features
        x = (x - offset_x) / scale_x

        # normalize observed features
        x0 = (s_0[use_ixs] - offset_x) / scale_x

        # normalize labels
        y = parameters
        mean_y = np.mean(y, axis=0)
        std_y = np.std(y, axis=0)
        y = (y - mean_y) / std_y

        return {
            "x": x, "y": y, "weights": weights, "use_ixs": use_ixs, "x0": x0,
        }

    @staticmethod
    def calculate_sensis(
        predictor: Predictor,
        fd_deltas: Union[List[float], float],
        x0: np.ndarray,
        n_x: int,
        n_y: int,
        par_trafo: ParTrafoBase,
        normalize_by_par: bool,
    ):
        """Calculate normalized predictor sensitivities.

        Parameters
        ----------
        predictor: Fitted predictor model.
        fd_deltas: Finite difference step sizes.
        x0: Observed data, shape (n_x).
        n_x: Data dimension.
        n_y: Transformed parameter dimension.
        par_trafo: Parameter transformations, shape (n_y).
        normalize_by_par: Whether to normalize sensitivities by parameters.

        Returns
        -------
        sensis: Sensitivities, shape (n_x, n_y).
        """
        def fun(_x):
            """Predictor function."""
            return predictor.predict(_x.reshape(1, -1)).flatten()

        # calculate sensitivities
        #  shape (n_x, n_y)
        sensis = fd_nabla1_multi_delta(
            x=x0, fun=fun, test_deltas=fd_deltas)
        if sensis.shape != (n_x, n_y):
            raise AssertionError("Sensitivity shape did not match.")

        # we are only interested in absolute values
        sensis = np.abs(sensis)

        # total sensitivities per parameter
        sensi_per_y = np.sum(sensis, axis=0)

        # identify parameters that have mostly zero gradients throughout
        y_has_sensi = ~np.isclose(sensi_per_y, 0.)
        # set values of near-zero contribution to zero
        sensis[:, ~y_has_sensi] = 0
        # log
        if not y_has_sensi.all():
            par_trafo_ids = par_trafo.get_ids()
            insensi_par_keys = [
                par_trafo_ids[ix] for ix in np.flatnonzero(~y_has_sensi)]
            logger.info(f"Zero info for parameters {insensi_par_keys}")

        if normalize_by_par:
            # normalize sums over sumstats to 1
            sensis[:, y_has_sensi] /= sensi_per_y[y_has_sensi]

        return sensis

    def get_weights(self, t: int) -> np.ndarray:
        info_weights: np.ndarray = \
            PNormDistance.for_t_or_latest(self.info_weights, t)
        return super().get_weights(t=t) * info_weights

    def requires_calibration(self) -> bool:
        if self.initial_info_weights is None:
            return True
        return self.sumstat.requires_calibration()

    def is_adaptive(self) -> bool:
        if self.fit_info_ixs.probably_has_late_events():
            return True
        return super().is_adaptive()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "predictor": self.predictor.__str__(),
            "fit_info_ixs": self.fit_info_ixs.__repr__(),
            "scale_function": self.scale_function.__name__,
            "max_info_weight_ratio": self.max_info_weight_ratio,
        })
        return config


class AggregatedDistance(Distance):
    """
    Aggregates a list of distance functions, all of which may work on subparts
    of the summary statistics. Then computes and returns the weighted sum of
    the distance values generated by the various distance functions.

    All class functions are propagated to the children and the obtained
    results aggregated appropriately.
    """

    def __init__(
        self,
        distances: List[Distance],
        weights: Union[List, dict] = None,
        factors: Union[List, dict] = None,
    ):
        """
        Parameters
        ----------

        distances: List
            The distance functions to apply.
        weights: Union[List, dict], optional (default = [1,...])
            The weights to apply to the distances when taking the sum. Can be
            a list with entries in the same order as the distances, or a
            dictionary of lists, with the keys being the single time points
            (if the weights should be iteration-specific).
        factors: Union[List, dict], optional (dfault = [1,...])
            Scaling factors that the weights are multiplied with. The same
            structure applies as to weights.
            If None is passed, a factor of 1 is considered for every summary
            statistic.
            Note that in this class, factors are superfluous as everything can
            be achieved with weights alone, however in subclasses the factors
            can remain static while weights adapt over time, allowing for
            greater flexibility.
        """
        super().__init__()

        if not isinstance(distances, list):
            distances = [distances]
        self.distances: List[Distance] = \
            [to_distance(distance) for distance in distances]

        self.weights = weights
        self.factors = factors

    def requires_calibration(self) -> bool:
        return any(d.requires_calibration() for d in self.distances)

    def is_adaptive(self) -> bool:
        return any(d.is_adaptive() for d in self.distances)

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ):
        super().initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )
        for distance in self.distances:
            distance.initialize(
                t=t,
                get_sample=get_sample,
                x_0=x_0,
                total_sims=total_sims,
            )
        self.format_weights_and_factors(t)

    def configure_sampler(
        self,
        sampler,
    ):
        """
        Note: `configure_sampler` is applied by all distances sequentially,
        so care must be taken that they perform no contradictory operations
        on the sampler.
        """
        for distance in self.distances:
            distance.configure_sampler(sampler)

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        total_sims: int,
    ) -> bool:
        """
        The `sum_stats` are passed on to all distance functions, each of
        which may then update using these. If any update occurred, a value
        of True is returned indicating that e.g. the distance may need to
        be recalculated since the underlying distances changed.
        """
        return any(distance.update(
            t=t,
            get_sample=get_sample,
            total_sims=total_sims,
        ) for distance in self.distances)

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None,
    ) -> float:
        """
        Applies all distance functions and computes the weighted sum of all
        obtained values.
        """
        values = np.array([
            distance(x, x_0, t, par) for distance in self.distances
        ])
        self.format_weights_and_factors(t)
        weights = AggregatedDistance.get_for_t_or_latest(self.weights, t)
        factors = AggregatedDistance.get_for_t_or_latest(self.factors, t)
        return float(np.dot(weights * factors, values))

    def get_config(self) -> dict:
        """
        Return configuration of the distance.

        Returns
        -------

        config: dict
            Dictionary describing the distance.
        """
        config = {}
        for j, distance in enumerate(self.distances):
            config[f'Distance_{j}'] = distance.get_config()
        return config

    def format_weights_and_factors(self, t):
        self.weights = AggregatedDistance.format_dict(
            self.weights, t, len(self.distances))
        self.factors = AggregatedDistance.format_dict(
            self.factors, t, len(self.distances))

    @staticmethod
    def format_dict(w, t, n_distances, default_val=1.):
        """
        Normalize weight or factor dictionary to the employed format.
        """
        if w is None:
            # use default
            w = {t: default_val * np.ones(n_distances)}
        elif not isinstance(w, dict):
            # f is not time-dependent
            # so just create one for time t
            w = {t: np.array(w)}
        return w

    @staticmethod
    def get_for_t_or_latest(w, t):
        """
        Extract values from dict for given time point.
        """
        # take last time point for which values exist
        if t not in w:
            t = max(w)
        # extract values for time point
        return w[t]


class AdaptiveAggregatedDistance(AggregatedDistance):
    """
    Adapt the weights of `AggregatedDistances` automatically over time.

    Parameters
    ----------
    distances:
        As in AggregatedDistance.
    initial_weights:
        Weights to be used in the initial iteration. List with
        a weight for each distance function.
    factors:
        As in AggregatedDistance.
    adaptive:
        True: Adapt weights after each iteration.
        False: Adapt weights only once at the beginning in initialize().
        This corresponds to a pre-calibration.
    scale_function:
        Function that takes a np.ndarray of shape (n_sample,),
        namely the values obtained by applying one of the distances on a set
        of samples, and returns a single float, namely the weight to apply to
        this distance function. Default: span.
    log_file:
        A log file to store weights for each time point in. Weights are
        currently not stored in the database. The data are saved in json
        format and can be retrieved via `pyabc.storage.load_dict_from_json`.
    """

    def __init__(
        self,
        distances: List[Distance],
        initial_weights: List = None,
        factors: Union[List, dict] = None,
        adaptive: bool = True,
        scale_function: Callable = None,
        log_file: str = None,
    ):
        super().__init__(distances=distances)
        self.initial_weights: List = initial_weights
        self.factors: Union[List, dict] = factors
        self.adaptive: bool = adaptive
        self.x_0: Union[dict, None] = None
        if scale_function is None:
            scale_function = span
        self.scale_function: Callable = scale_function
        self.log_file: str = log_file

    def requires_calibration(self) -> bool:
        return (self.initial_weights is None
                or any(d.requires_calibration() for d in self.distances))

    def is_adaptive(self) -> bool:
        return (self.adaptive
                or any(d.is_adaptive() for d in self.distances))

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ):
        """
        Initialize weights.
        """
        super().initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )
        self.x_0 = x_0

        if self.initial_weights is not None:
            self.weights[t] = self.initial_weights
            return

        # execute function
        sample = get_sample()

        # update weights from samples
        self._update(t, sample)

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        total_sims: int,
    ):
        """
        Update weights based on all simulations.
        """
        super().update(t=t, get_sample=get_sample, total_sims=total_sims)

        if not self.adaptive:
            return False

        # execute function
        sample = get_sample()

        self._update(t, sample)

        return True

    def _update(
        self,
        t: int,
        sample: Sample,
    ):
        """
        Here the real update of weights happens.
        """
        # to-be-filled-and-appended weights dictionary
        w = []

        sum_stats = sample.all_sum_stats

        for distance in self.distances:
            # apply distance to all samples
            current_list = np.array([
                distance(sum_stat, self.x_0)
                for sum_stat in sum_stats
            ])
            # compute scaling
            scale = self.scale_function(samples=current_list)

            # compute weight (inverted scale)
            if np.isclose(scale, 0):
                # This means that either the summary statistic is not in the
                # samples, or that all simulations were identical. In either
                # case, it should be safe to ignore this summary statistic.
                w.append(0)
            else:
                w.append(1 / scale)

        w = np.array(w)
        if w.size != len(self.distances):
            raise AssertionError(
                f"weights.size={w.size} != "
                f"len(distances)={len(self.distances)}")

        # add to w attribute, at time t
        self.weights[t] = np.array(w)

        # logging
        self.log(t)

    def configure_sampler(self, sampler) -> None:
        """
        Make the sampler return also rejected particles,
        because these are needed to get a better estimate of the summary
        statistic variability, avoiding a bias to accepted ones only.

        Parameters
        ----------

        sampler: Sampler
            The sampler employed.
        """
        super().configure_sampler(sampler=sampler)
        if self.adaptive:
            sampler.sample_factory.record_rejected()

    def log(self, t: int) -> None:
        logger.debug(f"Weights[{t}] = {self.weights[t]}")

        if self.log_file:
            save_dict_to_json(self.weights, self.log_file)


class DistanceWithMeasureList(Distance, ABC):
    """
    Base class for distance functions with measure list.
    This class is not functional on its own.

    Parameters
    ----------

    measures_to_use: Union[str, List[str]].
        * If set to "all", all measures are used. This is the default.
        * If a list is provided, the measures in the list are used.
        * measures refers to the summary statistics.
    """

    def __init__(
        self,
        measures_to_use='all',
    ):
        super().__init__()
        # the measures (summary statistics) to use for distance calculation
        self.measures_to_use = measures_to_use

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ):
        if self.measures_to_use == 'all':
            self.measures_to_use = x_0.keys()

    def get_config(self):
        config = super().get_config()
        config["measures_to_use"] = self.measures_to_use
        return config


class ZScoreDistance(DistanceWithMeasureList):
    """
    Calculate distance as sum of ZScore over the selected measures.
    The measured Data is the reference for the ZScore.

    Hence

    .. math::
        d(x, y) = \
        \\sum_{i \\in \\text{measures}} \\left| \\frac{x_i-y_i}{y_i} \\right|
    """

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None,
    ) -> float:
        return sum(abs((x[key] - x_0[key]) / x_0[key]) if x_0[key] != 0 else
                   (0 if x[key] == 0 else np.inf)
                   for key in self.measures_to_use) / len(self.measures_to_use)


class PCADistance(DistanceWithMeasureList):
    """
    Calculate distance in whitened coordinates.

    A whitening transformation :math:`X` is calculated from an initial sample.
    The distance is measured as euclidean distance in the transformed space.
    I.e

    .. math::

        d(x,y) = \\| Wx - Wy \\|
    """

    def __init__(self, measures_to_use='all'):
        super().__init__(measures_to_use)
        self._whitening_transformation_matrix = None

    def _dict_to_vect(self, x):
        return np.asarray([x[key] for key in self.measures_to_use])

    def _calculate_whitening_transformation_matrix(self, sum_stats):
        samples_vec = np.asarray([self._dict_to_vect(x)
                                  for x in sum_stats])
        # samples_vec is an array of shape nr_samples x nr_features
        means = samples_vec.mean(axis=0)
        centered = samples_vec - means
        covariance = centered.T.dot(centered)
        w, v = la.eigh(covariance)
        self._whitening_transformation_matrix = (
            v.dot(np.diag(1. / np.sqrt(w))).dot(v.T))

    def requires_calibration(self) -> bool:
        return True

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ):
        super().initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )

        # execute function
        all_sum_stats = get_sample().all_sum_stats

        self._calculate_whitening_transformation_matrix(all_sum_stats)

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None,
    ) -> float:
        x_vec, x_0_vec = self._dict_to_vect(x), self._dict_to_vect(x_0)
        distance = la.norm(
            self._whitening_transformation_matrix.dot(x_vec - x_0_vec), 2)
        return distance


class RangeEstimatorDistance(DistanceWithMeasureList):
    """
    Abstract base class for distance functions which estimate is based on a
    range.

    It defines the two template methods ``lower`` and ``upper``.

    Hence

    .. math::

        d(x, y) = \
        \\sum_{i \\in \\text{measures}} \\left | \\frac{x_i - y_i}{u_i - l_i}\
          \\right |

    where :math:`l_i` and :math:`u_i` are the lower and upper
    margin for measure :math:`i`.
    """

    @staticmethod
    def lower(parameter_list: List[float]):
        """
        Calculate the lower margin form a list of parameter values.

        Parameters
        ----------

        parameter_list: List[float]
            List of values of a parameter.

        Returns
        -------

        lower_margin: float
            The lower margin of the range calculated from these parameters
        """

    @staticmethod
    def upper(parameter_list: List[float]):
        """
        Calculate the upper margin form a list of parameter values.

        Parameters
        ----------

        parameter_list: List[float]
            List of values of a parameter.

        Returns
        -------

        upper_margin: float
            The upper margin of the range calculated from these parameters
        """

    def __init__(self, measures_to_use='all'):
        super().__init__(measures_to_use)
        self.normalization = None

    def get_config(self):
        config = super().get_config()
        config["normalization"] = self.normalization
        return config

    def _calculate_normalization(self, sum_stats):
        measures = {name: [] for name in self.measures_to_use}
        for sample in sum_stats:
            for measure in self.measures_to_use:
                measures[measure].append(sample[measure])
        self.normalization = {measure:
                              self.upper(measures[measure])
                              - self.lower(measures[measure])
                              for measure in self.measures_to_use}

    def requires_calibration(self) -> bool:
        return True

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ):
        super().initialize(
            t=t,
            get_sample=get_sample,
            x_0=x_0,
            total_sims=total_sims,
        )

        # execute function
        all_sum_stats = get_sample().all_sum_stats

        self._calculate_normalization(all_sum_stats)

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None,
    ) -> float:
        distance = sum(abs((x[key] - x_0[key]) / self.normalization[key])
                       for key in self.measures_to_use)
        return distance


class MinMaxDistance(RangeEstimatorDistance):
    """
    Calculate upper and lower margins as max and min of the parameters.
    This works surprisingly well for normalization in simple cases
    """

    @staticmethod
    def upper(parameter_list):
        return max(parameter_list)

    @staticmethod
    def lower(parameter_list):
        return min(parameter_list)


class PercentileDistance(RangeEstimatorDistance):
    """
    Calculate normalization 20% and 80% from percentiles as lower
    and upper margins
    """

    PERCENTILE = 20  #: The percentiles

    @staticmethod
    def upper(parameter_list):
        return np.percentile(parameter_list,
                             100 - PercentileDistance.PERCENTILE)

    @staticmethod
    def lower(parameter_list):
        return np.percentile(parameter_list,
                             PercentileDistance.PERCENTILE)

    def get_config(self):
        config = super().get_config()
        config["PERCENTILE"] = self.PERCENTILE
        return config
