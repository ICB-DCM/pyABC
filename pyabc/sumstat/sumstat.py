"""Summary statistics learning."""

import numpy as np
from typing import Callable, Dict, List, Union
import copy
from abc import ABC, abstractmethod

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
except ImportError as e:
    LinearRegression = GaussianProcessRegressor = DotProduct = WhiteKernel \
        = None

from ..population import Sample
from .util import io_dict2arr, dict2arrlabels


class Sumstat(ABC):
    """Summary statistics."""

    def __init__(self, prev: 'Sumstat' = None):
        """
        Parameters
        ----------
        prev: Previously applied summary statistics, enables chaining.
        """
        # data keys (for correct order)
        self.x_keys: Union[List[str], None] = None
        self.x_0: Union[dict, None] = None
        self.prev: Union['Sumstat', None] = prev

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
        sumstat: Summary statistics of the data, an ndarray.
        """

    def initialize(
            self,
            t: int,
            get_sample: Callable[[], Sample],
            x_0: dict = None) -> None:
        """Initialize before the first generation.

        Called at the beginning by the inference routine, can be used for
        calibration to the problem.
        The default is to do nothing.

        Parameters
        ----------
        t:
            Time point for which to initialize the distance.
        get_sample:
            Returns on command the initial sample.
        x_0:
            The observed summary statistics.
        """
        self.x_keys: List[str] = list(x_0.keys())
        self.x_0: dict = x_0
        if self.prev is not None:
            self.prev.initialize(t=t, get_sample=get_sample, x_0=x_0)

    def update(
            self,
            t: int,
            get_sample: Callable[[], Sample]) -> bool:
        """Update for the upcoming generation t.

        Similar as `initialize`, however called for every subsequent iteration.
        The default is to do nothing.

        Parameters
        ----------
        t:
            Time point for which to update the distance.
        get_sample:
            Returns on demand the last generation's complete sample.

        Returns
        -------
        is_updated: bool
            Whether something has changed compared to beforehand.
            Depending on the result, the population needs to be updated
            before preparing the next generation.
            Defaults to False.
        """
        if self.prev is not None:
            return self.prev.update(t=t, get_sample=get_sample)
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
        if self.prev is not None:
            self.prev.configure_sampler(sampler=sampler)

    def requires_calibration(self) -> bool:
        """
        Whether the class requires an initial calibration, based on
        samples from the prior. Default: False.
        """
        if self.prev is not None:
            return self.prev.requires_calibration()
        return False

    def is_adaptive(self) -> bool:
        """
        Whether the class is dynamically updated after each generation,
        based on the last generation's available data. Default: False.
        """
        if self.prev is not None:
            return self.prev.is_adaptive()
        return False

    def get_ids(self) -> List[str]:
        """Get ids/labels for the summary statistics."""
        # default: indices
        s_0 = self(self.x_0)
        return [f"S_{ix}" for ix in range(s_0.size)]


class IdentitySumstat(Sumstat):
    """Identity mapping with optional transformation to apply."""

    def __init__(
            self,
            trafos: List[Callable[[np.ndarray], np.ndarray]] = None,
            prev: Sumstat = None):
        """
        Parameters
        ----------
        prev:
            Previously applied summary statistics, enables chaining.
        trafos:
            Optional transformations to apply, should be vectorized.
            Note that if the original data should be contained, a
            transformation s.a. `lambda x: x` must be added.
        """
        super().__init__(prev=prev)
        self.trafos = trafos

    @io_dict2arr
    def __call__(self, data: Union[dict, np.ndarray]) -> np.ndarray:
        # apply previous statistics
        if self.prev is not None:
            data = self.prev(data)
        # apply transformations
        if self.trafos is not None:
            data = np.asarray([trafo(data) for trafo in self.trafos])
        return data

    def get_ids(self):
        """Get ids/labels for the summary statistics.

        Uses the more meaningful data labels if the transformation is id.
        """
        if self.prev is None and self.trafos is None:
            return dict2arrlabels(self.x_0, keys=self.x_keys)
        return super().get_ids()


class PredictorSumstat(Sumstat):
    """Summary statistics based on predictors targeting parameters, y -> theta.

    Based on [#fearnheadprangle]_ with general predictors.

    The predictor should define:
    * `fit(X, y)` to fit the model on a sample of data X and outputs y,
      where X has shape (n_sample, n_in), with n_in the number of input
      features, and
      y has shape (n_sample, n_out), with n_out either the parameter dimension
      or 1, depending on `joint`.
    * `predict(X)` to predict outputs, where X has shape (n_sample, n_in).

    .. [#fearnheadprangle]
        Fearnhead, Paul, and Dennis Prangle.
        "Constructing summary statistics for approximate Bayesian computation:
        Semi‐automatic approximate Bayesian computation."
        Journal of the Royal Statistical Society: Series B
        (Statistical Methodology) 74.3 (2012): 419-474.
    """

    def __init__(
            self,
            predictor,
            adaptive: bool = False,
            joint: bool = True,
            prev: Sumstat = None):
        """
        Parameters
        ----------
        predictor:
            The predictor mapping data (inputs) to parameters (targets). See
            above for the class contract.
        adaptive:
            False: Only train the model at the beginning in `initialize`.
            True: Update the model also after each generation in `update`.
        joint:
            Whether the predictor learns one model for all parameters, or
            separate models per parameter.
        prev:
            Previously applied summary statistics, enables chaining.
        """
        if prev is None:
            prev = IdentitySumstat()
        super().__init__(prev=prev)
        self.predictor = predictor
        # used if parameters are predicted separately
        self.single_predictors = None
        self.adaptive = adaptive
        self.joint = joint

        # parameter keys (for correct order)
        self.par_keys = None

    def initialize(
            self,
            t: int,
            get_sample: Callable[[], Sample],
            x_0: dict = None) -> None:
        super().initialize(t=t, get_sample=get_sample, x_0=x_0)

        sample = get_sample()

        # fix key order
        self.par_keys: List[str] = \
            list(sample.accepted_particles[0].parameter.keys())

        # copy predictors if not joint
        if not self.joint:
            self.single_predictors = [
                copy.deepcopy(self.predictor) for _ in self.par_keys]

        # fit model to sample
        self.fit(sample)

    def update(
            self,
            t: int,
            get_sample: Callable[[], Sample]) -> bool:
        updated = super().update(t, get_sample)
        if not self.adaptive:
            return updated

        # fit model to sample
        sample = get_sample()
        self.fit(sample)
        return True

    def configure_sampler(self, sampler) -> None:
        if self.adaptive:
            # record rejected particles
            sampler.sample_factory.record_rejected = True

    def requires_calibration(self) -> bool:
        return True

    def is_adaptive(self) -> bool:
        if self.adaptive:
            return True
        if self.prev is not None:
            return self.prev.is_adaptive()
        return False

    def fit(self, sample: Sample) -> None:
        """Fit the model to the sample.

        Parameters
        ----------
        sample: Calibration or last generation's sample.
        """
        # TODO use weights?
        sumstats, pars, _ = self.read_sample(sample)

        if self.joint:
            self.predictor.fit(sumstats, pars)
        else:
            for predictor, par in zip(self.single_predictors, pars.T):
                predictor.fit(sumstats, par)



    @io_dict2arr
    def __call__(self, data: Union[dict, np.ndarray]):
        data = self.prev(data)
        return self.predictor.predict(data.reshape(1, -1))


class LinearPredictorSumstat(PredictorSumstat):
    """Use a linear predictor for the mapping y -> theta.

    This implementation is based on and requires an installation of sklearn,
    install via `pip install pyabc[scikit-learn]`.
    """

    def __init__(
            self,
            adaptive: bool = False,
            joint: bool = True,
            prev: Sumstat = None,
    ):
        # check installation
        if LinearRegression is None:
            raise ImportError(
                "This sumstat requires an installation of sklearn. Install"
                "e.g. via `pip install pyabc[scikit-learn]`")

        super().__init__(
            LinearRegression(fit_intercept=True),
            adaptive=adaptive,
            joint=joint,
            prev=prev,
        )

    def fit(self, sample: Sample):
        sumstats, parameters, weights = self.read_sample(sample)

        # z-score normalization
        self.mean = np.mean(sumstats, axis=0)
        self.std = np.std(sumstats, axis=0)
        sumstats = (sumstats - self.mean) / self.std

        self.predictor.fit(X=sum_stats, y=parameters, sample_weight=weights)

    @io_dict2arr
    def __call__(self, data):
        data = (data - self.mean) / self.std
        return self.predictor.predict(data.reshape(1, -1))


class GPBritney(PredictorSumstat):

    def __init__(self, kernel=None, adaptive: bool = False, joint: bool = True):
        if kernel is None:
            kernel = DotProduct() + WhiteKernel()
        gp_predictor = GaussianProcessRegressor(kernel=kernel)
        super().__init__(predictor=gp_predictor, adaptive=adaptive, joint=joint)


def read_sample(
        sample: Sample, prev: Sumstat):
    """Read in sample.

    Parameters
    ----------
    sample: Calibration or last generation's sample.

    Returns
    -------
    sumstats, parameters, weights:
        Arrays of shape (n_sample, n_out).
    """
    # use all particles
    # TODO allow to use only accepted ones?
    particles = sample.all_particles

    # dimensions of sample, summary statistics, and parameters
    n_sample = len(particles)
    n_sumstat = len(prev(particles[0].sum_stat))
    n_par = len(particles[0].parameter)

    # prepare matrices
    sumstats = np.empty((n_sample, n_sumstat))
    parameters = np.empty((n_sample, n_par))
    weights = np.empty((n_sample,))

    # fill by iteration over all particles
    for i_particle, particle in enumerate(particles):
        sumstats[i_particle, :] = prev(particle.sum_stat)
        parameters[i_particle, :] = prev(particle.parameter)
        weights[i_particle] = particle.weight
    return sumstats, parameters, weights
