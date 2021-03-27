"""Distance functions."""
from abc import ABC

import numpy as np
from scipy import linalg as la
from numbers import Number
from typing import Dict, List, Callable, Union
import logging

from .scale import standard_deviation, span
from .base import Distance, to_distance
from .util import bound_weights, log_weights
from ..storage import save_dict_to_json
from ..population import Sample
from ..predictor import Predictor
from ..sumstat import Sumstat, IdentitySumstat
from ..sumstat.util import dict2arr, read_sample


logger = logging.getLogger("Distance")


class PNormDistance(Distance):
    """Weighted p-norm distance.

    Distance between summary statistics calculated according to

    .. math::

        d(x, y) = \
        \\left [\\sum_{i} \\left| w_i ( x_i-y_i ) \\right|^{p} \\right ]^{1/p}

    E.g.
    * p=1 for a Euclidean or L1 metric,
    * p=2 for a Manhattan or L2 metric,
    * p=np.inf for a Chebyshev, maximum or Linf metric.

    Parameters
    ----------
    p:
        p for p-norm. p >= 1, p = np.inf implies max norm.
    weights:
        Weights. Dictionary indexed by time points, or only one entry if
        the weights should not change over time.
        Each entry contains a dictionary of numeric weights, indexed by summary
        statistics labels, or the corresponding array representation.
        If None is passed, a weight of 1 is considered for every summary
        statistic.
        If no entry is available in `weights` for a given time point, the
        maximum available time point is selected.
    factors:
        Fixed multiplicative factors the weights are multiplied with. The
        discrimination of weights and factors makes only sense for adaptive
        distance functions.
    sumstat:
        Summary statistics transformation to apply to the model output.
    """

    def __init__(
        self,
        p: float = 2,
        weights: Union[Dict[str, float], Dict[int, Dict[str, float]]] = None,
        factors: Dict[str, float] = None,
        sumstat: Sumstat = None,
    ):
        super().__init__()

        if sumstat is None:
            sumstat = IdentitySumstat()
        self.sumstat: Sumstat = sumstat

        if p < 1:
            raise ValueError("It must be p >= 1")
        self.p: float = p

        self._arg_weights = weights
        self._arg_factors = factors

        self.weights: Union[Dict[int, np.ndarray], None] = None
        self.factors: Union[np.ndarray, None] = None

        # to cache the observed data and summary statistics
        self.x_0: Union[dict, None] = None
        self.s_0: Union[np.ndarray, None] = None

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict = None
    ) -> None:
        # update summary statistics
        self.sumstat.initialize(t=t, get_sample=get_sample, x_0=x_0)

        # observed data
        self.x_0 = x_0
        self.s_0 = self.sumstat(self.x_0)

        # initialize weights
        self.weights = PNormDistance.format_dict(
            vals=self._arg_weights, t=t, s_0=self.s_0,
            s_ids=self.sumstat.get_ids(),
        )
        self.factors = PNormDistance.format_dict(
            vals=self._arg_factors, t=t, s_0=self.s_0,
            s_ids=self.sumstat.get_ids(),
        )[t]

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample],
    ) -> bool:
        # update summary statistics
        updated = self.sumstat.update(t=t, get_sample=get_sample)
        if updated:
            self.s_0 = self.sumstat(self.x_0)
        return updated

    def configure_sampler(self, sampler) -> None:
        self.sumstat.configure_sampler(sampler=sampler)

    @staticmethod
    def format_dict(
        vals: Union[Dict[str, float], Dict[int, Dict[str, float]]],
        t: int,
        s_0: np.ndarray,
        s_ids: List[str],
        default_val: float = 1.
    ) -> Dict[int, np.ndarray]:
        """Normalize weight dictionary to the employed format."""
        if vals is None:
            # use default
            vals = {t: default_val * np.ones(len(s_0))}
        elif isinstance(next(iter(vals.values())), Number):
            vals = {t: vals}

        # convert dicts to arrays
        for _t, dct in vals.items():
            vals[_t] = dict2arr(dct, keys=s_ids)

        return vals

    @staticmethod
    def for_t_or_latest(w: Dict[int, np.ndarray], t: int) -> np.ndarray:
        """
        Extract values from dict for given time point.
        """
        # take last time point for which values exist
        if t not in w:
            t = max(w.keys())
        # extract values for time point
        return w[t]

    def __call__(
        self,
        x: dict,
        x_0: dict,
        t: int = None,
        par: dict = None
    ) -> float:
        # extract weights for given time point
        weights = PNormDistance.for_t_or_latest(self.weights, t)
        factors = self.factors

        # compute summary statistics
        s, s0 = self.sumstat(x), self.sumstat(x_0)

        # assert shapes match
        if (
            s.shape != weights.shape or s.shape != s0.shape
            or s.shape != factors.shape
        ):
            raise AssertionError("Shapes do not match")

        # component-wise distances
        distances = np.abs(factors * weights * (s - s0))

        # maximum or p-norm distance
        if self.p == np.inf:
            return distances.max()
        return (distances**self.p).sum()**(1/self.p)

    def get_config(self) -> dict:
        return {
            "name": self.__class__.__name__,
            "p": self.p,
            "weights": self.weights,
            "factors": self.factors,
            "sumstat": self.sumstat.__str__(),
        }

    def get_weights_dict(self) -> Dict[int, Dict[str, float]]:
        """Create labeled weights dictionary."""
        # assumes that the summary statistic labels do not change over time
        return {
            t: {
                key: val
                for key, val in zip(self.sumstat.get_ids(), self.weights[t])
            }
            for t in self.weights
        }


class AdaptivePNormDistance(PNormDistance):
    """
    In the p-norm distance, adapt the weights for each generation, based on
    the previous simulations. This class is motivated by [#prangle]_.

    Parameters
    ----------
    p:
        p for p-norm. Required p >= 1, p = np.inf allowed (infinity-norm).
        Default: p=2.
    initial_weights:
        Weights to be used in the initial iteration. Dictionary with
        observables as keys and weights as values.
    factors:
        As in PNormDistance.
    adaptive:
        True: Adapt distance after each iteration.
        False: Adapt distance only once at the beginning in initialize().
        This corresponds to a pre-calibration.
    scale_function:
        (data: list, x_0: float) -> scale: float. Computes the scale (i.e.
        inverse weight s = 1 / w) for a given summary statistic. Here, data
        denotes the list of simulated summary statistics, and x_0 the observed
        summary statistic. Implemented are absolute_median_deviation,
        standard_deviation (default), centered_absolute_median_deviation,
        centered_standard_deviation.
    max_weight_ratio:
        If not None, large weights will be bounded by the ratio times the
        smallest non-zero absolute weight. In practice usually not necessary,
        it is theoretically required to ensure convergence.
    log_file:
        A log file to store weights for each time point in. Weights are
        currently not stored in the database. The data are saved in json
        format and can be retrieved via `pyabc.storage.load_dict_from_json`.


    .. [#prangle] Prangle, Dennis. "Adapting the ABC Distance Function".
                Bayesian Analysis, 2017. doi:10.1214/16-BA1002.
    """

    def __init__(
        self,
        p: float = 2,
        initial_weights: Dict[str, float] = None,
        factors: Dict[str, float] = None,
        adaptive: bool = True,
        scale_function: Callable = None,
        max_weight_ratio: float = None,
        log_file: str = None,
        sumstat: Sumstat = None,
    ):
        # call p-norm constructor
        super().__init__(p=p, weights=None, factors=factors, sumstat=sumstat)

        self.initial_weights: Dict[str, float] = initial_weights

        self.adaptive: bool = adaptive

        if scale_function is None:
            scale_function = standard_deviation
        self.scale_function: Callable = scale_function

        self.max_weight_ratio: float = max_weight_ratio
        self.log_file: str = log_file

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

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict = None
    ) -> None:
        # esp. initialize sumstats
        super().initialize(t=t, get_sample=get_sample, x_0=x_0)

        # are initial weights pre-defined
        if not self.requires_calibration():
            self.weights[t] = dict2arr(
                self.initial_weights, keys=self.sumstat.get_ids())
            return

        # execute cached function
        sample = get_sample()

        # update weights from samples
        self._fit(t=t, sample=sample)

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample]
    ) -> bool:
        # esp. updates summary statistics
        updated = super().update(t=t, get_sample=get_sample)
        if not self.is_adaptive():
            return updated

        # execute cached function
        sample = get_sample()

        self._fit(t=t, sample=sample)

        return True

    def _fit(
        self,
        t: int,
        sample: Sample
    ) -> None:
        """Here the real weight update happens."""
        # create (n_sample, n_feature) matrix of all summary statistics
        ss = np.array(
            [self.sumstat(p.sum_stat).flatten() for p in sample.all_particles]
        )

        # observed summary statistics
        s0 = self.s_0

        # calculate scales
        scales = self.scale_function(samples=ss, s0=s0)

        # weights are the inverse scales
        # a scale close to zero happens e.g. if all simulations are identical
        # in any case, it should be safe to ignore this statistic
        weights = np.zeros_like(scales)
        weights[~np.isclose(scales, 0)] = 1 / scales[~np.isclose(scales, 0)]

        if (weights < 0).any():
            raise AssertionError(f"There are weights <0: {weights}")

        # bound weights
        weights = bound_weights(
            weights, max_weight_ratio=self.max_weight_ratio)

        # update weights attribute
        self.weights[t] = weights

        # logging
        log_weights(
            t=t, weights=self.weights, keys=self.sumstat.get_ids(),
            log_file=self.log_file, logger=logger)

    def requires_calibration(self) -> bool:
        if self.initial_weights is None:
            return True
        return self.sumstat.requires_calibration()

    def is_adaptive(self) -> bool:
        if self.adaptive:
            return True
        return self.sumstat.is_adaptive()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "adaptive": self.adaptive,
            "scale_function": self.scale_function.__name__,
            "max_weight_ratio": self.max_weight_ratio,
        })
        return config


class InfoWeightedPNormDistance(PNormDistance):
    """Weight summary statistics by sensitivity of a predictor `y -> theta`."""

    def __init__(
            self,
            predictor: Predictor,
            p: float = 2,
            initial_weights: Dict[str, float] = None,
            factors: Dict[str, float] = None,
            n_fit: int = 1,
            scale_function: Callable = None,
            max_weight_ratio: float = None,
            log_file: str = None,
            sumstat: Sumstat = None,
            eps: float = 1e-2,
    ):
        # call p-norm constructor
        super().__init__(p=p, weights=None, factors=factors, sumstat=sumstat)

        self.predictor = predictor

        self.initial_weights: Dict[str, float] = initial_weights

        self.n_fit: int = n_fit

        if scale_function is None:
            scale_function = standard_deviation
        self.scale_function: Callable = scale_function

        self.max_weight_ratio: float = max_weight_ratio
        self.log_file: str = log_file
        self.eps: float = eps

        # parameter keys (for correct order)
        self.par_keys: Union[List[str], None] = None

        # fit counter
        self.i_fit: int = 0

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
        if self.n_fit > 1:
            sampler.sample_factory.record_rejected()

    def initialize(
            self,
            t: int,
            get_sample: Callable[[], Sample],
            x_0: dict = None
    ) -> None:
        # esp. initialize sumstats
        super().initialize(t=t, get_sample=get_sample, x_0=x_0)

        # are initial weights pre-defined
        if not self.requires_calibration():
            self.weights[t] = dict2arr(
                self.initial_weights, keys=self.sumstat.get_ids())
            return

        # execute cached function
        sample = get_sample()

        # fix parameter key order
        self.par_keys: List[str] = \
            list(sample.accepted_particles[0].parameter.keys())

        # check whether refitting is desired
        if self.i_fit >= self.n_fit:
            return
        else:
            self.i_fit += 1

        # update weights from samples
        self._fit(t=t, sample=sample)

    def update(
            self,
            t: int,
            get_sample: Callable[[], Sample]
    ) -> bool:
        # esp. updates summary statistics
        updated = super().update(t=t, get_sample=get_sample)

        # check whether refitting is desired
        if self.i_fit >= self.n_fit:
            return updated
        else:
            self.i_fit += 1

        # execute cached function
        sample = get_sample()

        self._fit(t=t, sample=sample)

        return True

    def _fit(
            self,
            t: int,
            sample: Sample
    ) -> None:
        """Here the real weight update happens."""
        # create (n_sample, n_feature) matrix of all summary statistics
        sumstats, parameters, weights = read_sample(
            sample=sample, sumstat=self.sumstat, all_particles=True,
            par_keys=self.par_keys,
        )
        s_0 = self.s_0

        # learn predictor model
        self.predictor.fit(x=sumstats, y=parameters, w=weights)

        # sensitivity matrix
        n_sumstat = sumstats.shape[1]
        n_par = parameters.shape[1]
        sensis = np.empty((n_sumstat, n_par))

        # calculate all sensitivities of the predictor
        eye = np.eye(n_sumstat)
        eps = self.eps
        for i_s in range(n_sumstat):
            vp = self.predictor.predict(
                (s_0 + eps * eye[i_s]).reshape(1, -1),
                orig_scale=False,
            )
            vm = self.predictor.predict(
                (s_0 - eps * eye[i_s]).reshape(1, -1),
                orig_scale=False,
            )
            sensis[i_s, :] = (vp - vm) / (2 * eps)

        # we are only interested in absolute values
        sensis = np.abs(sensis)

        # normalize sums over sumstats to 1
        sensis /= np.sum(sensis, axis=0)

        # the weight of a sumstat is the sum of the sensitivities over all
        #  parameters
        weights = np.sum(sensis, axis=1)

        # bound weights
        weights = bound_weights(
            weights, max_weight_ratio=self.max_weight_ratio)

        # update weights attribute
        self.weights[t] = weights

        # logging
        log_weights(
            t=t, weights=self.weights, keys=self.sumstat.get_ids(),
            log_file=self.log_file, logger=logger)

    def requires_calibration(self) -> bool:
        if self.initial_weights is None:
            return True
        return self.sumstat.requires_calibration()

    def is_adaptive(self) -> bool:
        if self.n_fit > 1:
            return True
        return self.sumstat.is_adaptive()

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "predictor": self.predictor.__str__(),
            "n_fit": self.n_fit,
            "scale_function": self.scale_function.__name__,
            "max_weight_ratio": self.max_weight_ratio,
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
            factors: Union[List, dict] = None):
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
            be achieved with weights alone, however in subclsses the factors
            can remain static while weights adapt over time, allowing for
            greater flexibility.
        """
        super().__init__()

        if not isinstance(distances, list):
            distances = [distances]
        self.distances = [to_distance(distance) for distance in distances]

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
            x_0: dict = None):
        super().initialize(t, get_sample, x_0)
        for distance in self.distances:
            distance.initialize(t, get_sample, x_0)
        self.format_weights_and_factors(t)

    def configure_sampler(
            self,
            sampler):
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
            get_sample: Callable[[], Sample]) -> bool:
        """
        The `sum_stats` are passed on to all distance functions, each of
        which may then update using these. If any update occurred, a value
        of True is returned indicating that e.g. the distance may need to
        be recalculated since the underlying distances changed.
        """
        return any(distance.update(t, get_sample)
                   for distance in self.distances)

    def __call__(
            self,
            x: dict,
            x_0: dict,
            t: int = None,
            par: dict = None) -> float:
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
        Function that takes a list of floats, namely the values obtained
        by applying one of the distances passed to a set of samples,
        and returns a single float, namely the weight to apply to this
        distance function. Default: scale_span.
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
            log_file: str = None):
        super().__init__(distances=distances)
        self.initial_weights = initial_weights
        self.factors = factors
        self.adaptive = adaptive
        self.x_0 = None
        if scale_function is None:
            scale_function = span
        self.scale_function = scale_function
        self.log_file = log_file

    def requires_calibration(self) -> bool:
        return (self.initial_weights is None
                or any(d.requires_calibration() for d in self.distances))

    def is_adaptive(self) -> bool:
        return (self.adaptive
                or any(d.is_adaptive() for d in self.distances))

    def initialize(self,
                   t: int,
                   get_sample: Callable[[], Sample],
                   x_0: dict = None):
        """
        Initialize weights.
        """
        super().initialize(t, get_sample, x_0)
        self.x_0 = x_0

        if self.initial_weights is not None:
            self.weights[t] = self.initial_weights
            return

        # execute function
        sample = get_sample()

        # update weights from samples
        self._update(t, sample)

    def update(self,
               t: int,
               get_sample: Callable[[], Sample]):
        """
        Update weights based on all simulations.
        """
        super().update(t, get_sample)

        if not self.adaptive:
            return False

        # execute function
        sample = get_sample()

        self._update(t, sample)

        return True

    def _update(self,
                t: int,
                sample: Sample):
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

    def get_weights_dict(self) -> Dict[int, Dict[str, float]]:
        return self.weights


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

    def __init__(self,
                 measures_to_use='all'):
        super().__init__()
        # the measures (summary statistics) to use for distance calculation
        self.measures_to_use = measures_to_use

    def initialize(self,
                   t: int,
                   get_sample: Callable[[], Sample],
                   x_0: dict = None):
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

    def __call__(self,
                 x: dict,
                 x_0: dict,
                 t: int = None,
                 par: dict = None) -> float:
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

    def initialize(self,
                   t: int,
                   get_sample: Callable[[], Sample],
                   x_0: dict = None):
        super().initialize(t, get_sample, x_0)

        # execute function
        all_sum_stats = get_sample().all_sum_stats

        self._calculate_whitening_transformation_matrix(all_sum_stats)

    def __call__(self,
                 x: dict,
                 x_0: dict,
                 t: int = None,
                 par: dict = None) -> float:
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

    def initialize(self,
                   t: int,
                   get_sample: Callable[[], Sample],
                   x_0: dict = None):
        super().initialize(t, get_sample, x_0)

        # execute function
        all_sum_stats = get_sample().all_sum_stats

        self._calculate_normalization(all_sum_stats)

    def __call__(self,
                 x: dict,
                 x_0: dict,
                 t: int = None,
                 par: dict = None) -> float:
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
