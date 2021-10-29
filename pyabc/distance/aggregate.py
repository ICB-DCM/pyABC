"""Aggregated distances."""

import logging
from typing import Callable, List, Union

import numpy as np

from ..population import Sample
from ..storage import save_dict_to_json
from .base import Distance, to_distance
from .scale import span

logger = logging.getLogger("ABC.Distance")


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
        self.distances: List[Distance] = [
            to_distance(distance) for distance in distances
        ]

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
        return any(
            distance.update(
                t=t,
                get_sample=get_sample,
                total_sims=total_sims,
            )
            for distance in self.distances
        )

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
        values = np.array(
            [distance(x, x_0, t, par) for distance in self.distances]
        )
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
            self.weights, t, len(self.distances)
        )
        self.factors = AggregatedDistance.format_dict(
            self.factors, t, len(self.distances)
        )

    @staticmethod
    def format_dict(w, t, n_distances, default_val=1.0):
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
        return self.initial_weights is None or any(
            d.requires_calibration() for d in self.distances
        )

    def is_adaptive(self) -> bool:
        return self.adaptive or any(d.is_adaptive() for d in self.distances)

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
            current_list = np.array(
                [distance(sum_stat, self.x_0) for sum_stat in sum_stats]
            )
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
                f"len(distances)={len(self.distances)}"
            )

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
