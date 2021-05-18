"""
Various schemes to compute scales for the AdaptivePNormDistance and the
AdaptiveAggregatedDistance.

The functions take arguments `samples`, a np.ndarray of the sampled summary
statistics of shape (n_sample, n_feature), and `s0`, a np.ndarray of the
observed summary statistics, shape (n_feature,). They return a np.ndarray
`scales` of shape (n_feature,).

For usage with AdaptivePNormDistance, the following are recommended:
* standard_deviation
* root_mean_square_deviation (also takes observation into account)
* median_absolute_deviation
* combined_median_absolute_deviation (also takes observation into account)

Aside, the median can be replaced by the mean, and in total we have
in addition the following:
* mean_absolute_deviation
* combined_mean_absolute_deviation
* bias (only distance to observation)
* median_absolute_deviation_to_observation (only distance to observation)
* mean_absolute_deviation_to_observation (only distance to observation)
* standard_deviation_to_observation (only distance to observation)

Here, "only distance to observation" means that the in-sample variation
is not taken into account.

For AdaptiveAggregatedDistance, for which the samples are not summary statistic
values, but distance values, instead use either of
* span
* mean
* median
"""
import numpy as np
from typing import List
import logging

logger = logging.getLogger("Distance")


def check_io(fun):
    """Check input and output for their dimensions.

    Wrapper around scale functions.
    """
    def checked_fun(samples: np.ndarray, **kwargs):
        if "s0" in kwargs:
            if (samples.ndim == 1 and np.ndim(kwargs["s0"]) > 0) or (
                    samples.ndim > 1 and
                    samples.shape[1] != kwargs["s0"].shape[0]):
                raise AssertionError("Shape mismatch of samples and s0")
        scales: np.ndarray = fun(samples=samples, **kwargs)
        if (samples.ndim == 1 and np.ndim(scales) > 0) or (
                samples.ndim > 1 and scales.shape != (samples.shape[1],)):
            raise AssertionError("Shape mismatch of s0 and scales")
        return scales
    return checked_fun


def warn_obs_off(off_ixs: np.ndarray, s_ids: List[str]):
    """Raise warnings for features with high bias to the samples.

    Parameters
    ----------
    off_ixs: Indices of features with suspicious bias.
    s_ids: List of textual feature labels.
    """
    off_ixs = np.asarray(off_ixs, dtype=int)
    for ix in off_ixs:
        logger.warning(
            f"Feature {ix} ({s_ids[ix]}) has a high bias, down-weighting")


@check_io
def median_absolute_deviation(*, samples: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculate the sample `median absolute deviation (MAD)
    <https://en.wikipedia.org/wiki/Median_absolute_deviation/>`_
    from the median, defined as median(abs(samples - median(samples)).
    """
    mad = np.nanmedian(np.abs(samples - np.nanmedian(samples, axis=0)), axis=0)
    return mad


@check_io
def mean_absolute_deviation(*, samples: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculate the mean absolute deviation from the mean.
    """
    mad = np.nanmean(np.abs(samples - np.nanmean(samples, axis=0)), axis=0)
    return mad


@check_io
def standard_deviation(*, samples: np.ndarray, **kwargs) -> np.ndarray:
    """
    Calculate the sample `standard deviation (SD)
    <https://en.wikipedia.org/wiki/Standard_deviation/>`_.
    """
    std = np.nanstd(samples, axis=0)
    return std


std = standard_deviation


@check_io
def bias(*, samples: np.ndarray, s0: np.ndarray) -> np.ndarray:
    """Bias of sample to observed value."""
    bias = np.nanmean(samples, axis=0) - s0
    return bias


@check_io
def root_mean_square_deviation(
    *, samples: np.ndarray, s0: np.ndarray, s_ids: List[str],
) -> np.ndarray:
    """
    Square root of the mean squared error, i.e.
    of the bias squared plus the variance.
    """
    bs = bias(samples=samples, s0=s0)
    std = standard_deviation(samples=samples)
    mse = bs**2 + std**2
    rmse = np.sqrt(mse)

    # debugging
    warn_obs_off(off_ixs=np.flatnonzero(bs > 2 * std), s_ids=s_ids)

    return rmse


rmsd = root_mean_square_deviation


@check_io
def median_absolute_deviation_to_observation(
        *, samples: np.ndarray, s0: np.ndarray) -> np.ndarray:
    """Median absolute deviation of samples w.r.t. the observation s0."""
    mado = np.nanmedian(np.abs(samples - s0), axis=0)
    return mado


@check_io
def mean_absolute_deviation_to_observation(
        *, samples: np.ndarray, s0: np.ndarray) -> np.ndarray:
    """Mean absolute deviation of samples w.r.t. the observation s0."""
    mado = np.nanmean(np.abs(samples - s0), axis=0)
    return mado


@check_io
def combined_median_absolute_deviation(
    *, samples: np.ndarray, s0: np.ndarray, s_ids: List[str],
) -> np.ndarray:
    """
    Compute the sum of the median absolute deviations to the
    median of the samples and to the observed value.
    """
    mad = median_absolute_deviation(samples=samples)
    mado = median_absolute_deviation_to_observation(samples=samples, s0=s0)
    cmad = mad + mado

    # debugging
    warn_obs_off(off_ixs=np.flatnonzero(mado > 2 * mad), s_ids=s_ids)

    return cmad


@check_io
def combined_mean_absolute_deviation(
    *, samples: np.ndarray, s0: np.ndarray, s_ids: List[str],
) -> np.ndarray:
    """
    Compute the sum of the mean absolute deviations to the
    mean of the samples and to the observed value.
    """
    mad = mean_absolute_deviation(samples=samples)
    mado = mean_absolute_deviation_to_observation(samples=samples, s0=s0)
    cmad = mad + mado

    # debugging
    warn_obs_off(off_ixs=np.flatnonzero(mado > 2 * mad), s_ids=s_ids)

    return cmad


@check_io
def standard_deviation_to_observation(
        *, samples: np.ndarray, s0: np.ndarray) -> np.ndarray:
    """
    Standard deviation of absolute deviations of the samples w.r.t.
    the observation s0.
    """
    stdo = np.nanstd(np.abs(samples - s0), axis=0)
    return stdo


@check_io
def span(*, samples: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the difference of largest and smallest sample point."""
    return np.nanmax(samples, axis=0) - np.nanmin(samples, axis=0)


@check_io
def mean(*, samples: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the mean."""
    return np.nanmean(samples, axis=0)


@check_io
def median(*, samples: np.ndarray, **kwargs) -> np.ndarray:
    """Compute the median."""
    return np.nanmedian(samples, axis=0)
