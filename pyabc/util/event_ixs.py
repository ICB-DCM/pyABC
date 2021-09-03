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
        sims: Union[Collection[int], int] = None,
        from_t: int = None,
        from_sims: int = None,
    ):
        """
        Parameters
        ----------
        ts:
            Time points at which something should happen. This can be either
            a collection of time points, or a single time point. A value of
            inf is interpreted as all time points.
        sims:
            Numbers of total simulations after which something should happen.
            This can be either a collection of total simulation numbers, or
            a single total simulation number.
        from_t:
            Always do something starting with generation `from_t`.
        from_sims:
            Always do something as soon as `from_sims` simulations have been
            hit.
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

        if sims is None:
            sims = []
        # convert single values to collection
        if not isinstance(sims, collections.abc.Collection):
            sims: Collection[int] = {sims}
        # check conversion to index
        for sim in sims:
            if int(sim) != sim:
                raise AssertionError(
                    f"Simulation number {sim} must be an int")
        self.sims: List[int] = list(sims)

        # track which simulation numbers have been hit
        self.sims_hit: List[bool] = [False] * len(self.sims)

        self.from_t: int = from_t
        self.from_sims: int = from_sims

    def act(
        self,
        t: int,
        total_sims: int,
        modify: bool = True,
    ) -> bool:
        """Inform whether to do something at a given time index `t`.

        .. note::
            This method is not idempotent regarding simulation counts, it
            should be called only once.

        Parameters
        ----------
        t: Time point
        total_sims: Total number of simulations so far.
        modify: Whether to remember actions. If False, can be safely re-called.

        Returns
        -------
        hit: Whether a criterion has been hit.
        """
        act = False

        # check time points
        if t in self.ts or np.inf in self.ts:
            act = True

        # check simulations
        for i_check_sims, (check_sims, hit) in enumerate(
                zip(self.sims, self.sims_hit)):
            if hit:
                continue
            if total_sims >= check_sims:
                act = True
                # record criterion to have been hit
                if modify:
                    self.sims_hit[i_check_sims] = True

        # check initial time point
        if self.from_t is not None and t >= self.from_t:
            act = True

        # check initial number of simulations
        if self.from_sims is not None and total_sims >= self.from_sims:
            act = True

        return act

    def probably_has_late_events(self) -> bool:
        """Whether event indices > 0 are likely to occur.

        This is useful in order to know whether to e.g. collect rejected
        particles.

        Returns
        -------
        True if indices > 0 are likely to evaluate to True, False otherwise.
        """
        return (
            any(t > 0 for t in self.ts) or len(self.sims) > 0
            or self.from_t is not None or self.from_sims is not None
        )

    def requires_calibration(self) -> bool:
        """Whether at time 0 an event is likely to happen.

        Returns
        -------
        Whether there will be a time-point event at time 0 (total simulations
        should typically noly occur later).
        """
        return (
            0 in self.ts or np.inf in self.ts
            or (self.from_t is not None and self.from_t == 0)
        )

    def __repr__(self) -> str:
        repr = f"<{self.__class__.__name__}"
        if self.ts:
            repr += f", ts={self.ts}"
        if self.sims:
            repr += f", sims={self.sims}"
        if self.from_t is not None:
            repr += f", from_t={self.from_t}"
        if self.from_sims is not None:
            repr += f", from_sims={self.from_sims}"
        repr += ">"
        return repr

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
