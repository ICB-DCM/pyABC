import numpy as np
from typing import Collection, List, Union
import collections.abc


class EventIxs:
    """Indicate whether iteration conditions apply for something to happen.

    Used to e.g. update weights, or train regression models.
    """

    def __init__(
        self,
        ts: Union[Collection[int], int] = None,
        total_sims: Union[Collection[int], int] = None,
    ):
        """
        Parameters
        ----------
        ts:
            Time points at which something should happen. This can be either
            a collection of time points, or a single time point. A value of
            inf is interpreted as all time points.
        total_sims:
            Numbers of total simulations after which something should happen.
            This can be either a collection of total simulation numbers, or
            a single total simulation number.
        """
        if ts is None:
            ts = []
        # convert single values to collection
        if not isinstance(ts, collections.abc.Collection):
            ts: Collection[int] = {ts}
        # check conversion to index
        for ix in ts:
            if ix != np.inf and int(ix) != ix:
                raise AssertionError(
                    f"Index {ix} must be either inf or an int")
        self.ts: Collection[int] = set(ts)

        if total_sims is None:
            total_sims = []
        # convert single values to collection
        if not isinstance(total_sims, collections.abc.Collection):
            total_sims: Collection[int] = {total_sims}
        # check conversion to index
        for total_sim in total_sims:
            if int(total_sim) != total_sim:
                raise AssertionError(
                    f"Simulation number {total_sims} must be an int")
        self.total_sims: List[int] = list(total_sims)

        # track which simulation numbers have been hit
        self.total_sims_hit: List[bool] = [False] * len(self.total_sims)

    def act(
        self,
        t: int,
        total_sims: int,
    ) -> bool:
        """Inform whether to do something at a given time index `t`.

        .. note::
            This method is not idempotent regarding simulation counts, it
            should be called only once.

        Parameters
        ----------
        t: Time point
        total_sims: Total number of simulations so far.

        Returns
        -------
        hit: Whether a criterion has been hit.
        """
        act = False

        # check time points
        if t in self.ts or np.inf in self.ts:
            act = True

        # check simulations
        for i_total_sim, (total_sim, hit) in enumerate(
                zip(self.total_sims, self.total_sims_hit)):
            if hit:
                continue
            if total_sims >= total_sim:
                act = True
                # record criterion to have been hit
                self.total_sims_hit[i_total_sim] = True

        return act

    def probably_has_late_events(self) -> bool:
        """Whether event indices > 0 are likely to occur.

        This is useful in order to know whether to e.g. collect rejected
        particles.

        Returns
        -------
        True if indices > 0 are likely to evaluate to True, False otherwise.
        """
        return any(t > 0 for t in self.ts) or len(self.total_sims) > 0

    def requires_calibration(self) -> bool:
        """Whether at time 0 an event is likely to happend.

        Returns
        -------
        Whether there will be a time-point event at time 0 (total simulations
        should typically noly occur later).
        """
        return 0 in self.ts or np.inf in self.ts

    def __repr__(self) -> str:
        return \
            f"<{self.__class__.__name__} ts={self.ts}, " \
            f"total_sims={self.total_sims}>"

    @staticmethod
    def to_instance(
        maybe_event_ixs: Union["EventIxs", Collection[int], int],
    ) -> "EventIxs":
        """Create instance from instance or collection of time points.

        Parameters
        ----------
        maybe_event_ixs:
            Can be either a `DoIx` already, or a collection of integers, or
            inf (which is interpreted as any time point), or an int (which
            is interpreted as `range(0, ..., maybe_do_ix)`), all of which
            are interpreted as time point criteria.

        Returns
        -------
        do_ix:
            A valid `DoIx` instance.
        """
        if isinstance(maybe_event_ixs, EventIxs):
            return maybe_event_ixs

        # otherwise we assume that time points were passed
        return EventIxs(ts=maybe_event_ixs)
