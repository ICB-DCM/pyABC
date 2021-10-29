"""Basic summary statistics."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from ..population import Sample
from ..util import dict2arrlabels, io_dict2arr


class Sumstat(ABC):
    """Summary statistics.

    Summary statistics operate on and transform the model output. They can e.g.
    rotate, augment, or extract features.
    Via the `pre` argument, summary statistics operations can be
    concatenated/chained.
    """

    def __init__(self, pre: 'Sumstat' = None):
        """
        Parameters
        ----------
        pre: Previously applied summary statistics, enables chaining.
        """
        # data keys (for correct order)
        self.x_keys: Union[List[str], None] = None
        # observed data
        self.x_0: Union[dict, None] = None
        # previous chained statistics
        self.pre: Union['Sumstat', None] = pre
        # ids
        self.ids: Union[List[str], None] = None

    @abstractmethod
    def __call__(
        self,
        data: Union[dict, np.ndarray],
    ) -> Union[np.ndarray, Dict[str, float]]:
        """Calculate summary statistics.

        Parameters
        ----------
        data: Model output or observed data.

        Returns
        -------
        sumstat: Summary statistics of the data, a np.ndarray.
        """

    def initialize(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        x_0: dict,
        total_sims: int,
    ) -> None:
        """Initialize before the first generation.

        Called at the beginning by the inference routine, can be used for
        calibration to the problem.

        Parameters
        ----------
        t:
            Time point for which to initialize the distance.
        get_sample:
            Returns on command the initial sample.
        x_0:
            The observed summary statistics.
        total_sims:
            The total number of simulations so far.
        """
        # record data keys
        self.x_keys: List[str] = list(x_0.keys())
        # record observed data
        self.x_0: dict = x_0
        # initialize previous statistics
        if self.pre is not None:
            self.pre.initialize(
                t=t, get_sample=get_sample, x_0=x_0, total_sims=total_sims
            )

    def update(
        self,
        t: int,
        get_sample: Callable[[], Sample],
        total_sims: int,
    ) -> bool:
        """Update for the upcoming generation t.

        Similar as `initialize`, however called for every subsequent iteration.

        Parameters
        ----------
        t:
            Time point for which to update the distance.
        get_sample:
            Returns on demand the last generation's complete sample.
        total_sims:
            The total number of simulations so far.

        Returns
        -------
        is_updated: bool
            Whether something has changed compared to beforehand.
            Depending on the result, the population needs to be updated
            before preparing the next generation.
            Defaults to False.
        """
        if self.pre is not None:
            return self.pre.update(
                t=t, get_sample=get_sample, total_sims=total_sims
            )
        return False

    def configure_sampler(self, sampler) -> None:
        """Configure the sampler.

        This method is called by the inference routine at the beginning.
        A possible configuration would be to request also the storing of
        rejected particles.
        The default is to do nothing.

        Parameters
        ----------
        sampler: Sampler
            The used sampler.
        """
        if self.pre is not None:
            self.pre.configure_sampler(sampler=sampler)

    def requires_calibration(self) -> bool:
        """
        Whether the class requires an initial calibration, based on
        samples from the prior. Default: False.
        """
        if self.pre is not None:
            return self.pre.requires_calibration()
        return False

    def is_adaptive(self) -> bool:
        """
        Whether the class is dynamically updated after each generation,
        based on the last generation's available data. Default: False.
        """
        if self.pre is not None:
            return self.pre.is_adaptive()
        return False

    def get_ids(self) -> List[str]:
        """Get ids/labels for the summary statistics.

        Defaults to indexing the statistics as `S_{ix}`.
        """
        s_0 = self(self.x_0)
        return [f"s_{ix}" for ix in range(s_0.size)]

    def __str__(self) -> str:
        return f"<{self.__class__.__name__} pre={self.pre.__str__()}>"


class IdentitySumstat(Sumstat):
    """Identity mapping with optional transformations."""

    def __init__(
        self,
        trafos: List[Callable[[np.ndarray], np.ndarray]] = None,
        pre: Sumstat = None,
        shape_out: Tuple[int, ...] = (-1,),
    ):
        """
        Parameters
        ----------
        pre:
            Previously applied summary statistics, enables chaining.
        trafos:
            Optional transformations to apply, should be vectorized.
            Note that if the original data should be contained, a
            transformation s.a. `lambda x: x` must be added.
        shape_out:
            Shape the (otherwise flat) output is converted to, via
            :py:func:`numpy.reshape`.
            Defaults to (-1,) and thus a flat array. Sometimes a row vector
            (1, -1) may be preferable, e.g. to treat simulations as replicates.
            For more complex shapes, tailored mappings may be preferable by
            deriving from Sumstat or IdentitySumstat.
        """
        super().__init__(pre=pre)
        self.trafos: List[Callable[[np.ndarray], np.ndarray]] = trafos
        self.shape_out: Tuple[int, ...] = shape_out

    @io_dict2arr
    def __call__(self, data: Union[dict, np.ndarray]) -> np.ndarray:
        """
        Returns
        -------
        sumstat: Concatenated summary statistics array of shape (n,1) or (1,n).
        """
        # apply previous statistics
        if self.pre is not None:
            data = self.pre(data)
        # apply transformations
        if self.trafos is not None:
            # create one long array until structure ever becomes interesting
            # also allows trafos to yield differing dimensions
            data = np.concatenate(
                [trafo(data).flatten() for trafo in self.trafos]
            )

        # reshape
        data = data.reshape(self.shape_out)

        return data

    def get_ids(self):
        """Get ids/labels for the summary statistics.

        Uses the more meaningful data labels if the transformation is id.
        """
        if self.pre is None and self.trafos is None:
            return dict2arrlabels(self.x_0, keys=self.x_keys)
        return super().get_ids()

    def __str__(self) -> str:
        return (
            f"<{self.__class__.__name__} pre={self.pre}, "
            f"trafos={self.trafos}>"
        )
