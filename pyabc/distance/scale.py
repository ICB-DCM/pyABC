"""
Various schemes to compute scales for the AdaptivePNormDistance and the
AdaptiveAggregatedDistance.

All of the functions take as argument a `samples` list of values, and some in
addition other arguments, in particular a float `s0` for the observed
value.

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


def median_absolute_deviation(samples, **kwargs):
    """
    Calculate the sample `median absolute deviation (MAD)
    <https://en.wikipedia.org/wiki/Median_absolute_deviation/>`_
    from the median, defined as
    median(abs(samples - median(samples)).
    """
    mad = np.nanmedian(np.abs(samples - np.nanmedian(samples, axis=0)), axis=0)
    return mad


def mean_absolute_deviation(samples, **kwargs):
    """
    Calculate the mean absolute deviation from the mean.
    """
    mad = np.nanmean(np.abs(samples - np.nanmean(samples, axis=0)), axis=0)
    return mad


def standard_deviation(samples, **kwargs):
    """
    Calculate the sample `standard deviation (SD)
    <https://en.wikipedia.org/wiki/Standard_deviation/>`_.
    """
    std = np.nanstd(samples, axis=0)
    return std


def bias(samples, s0, **kwargs):
    """
    Bias of sample to observed value.
    """
    bias = np.abs(np.nanmean(samples, axis=0) - s0)
    return bias


def root_mean_square_deviation(samples, s0, **kwargs):
    """
    Square root of the mean squared error, i.e.
    of the bias squared plus the variance.
    """
    bs = bias(samples, s0)
    std = standard_deviation(samples)
    mse = bs**2 + std**2
    rmse = np.sqrt(mse)
    return rmse


def median_absolute_deviation_to_observation(samples, s0, **kwargs):
    """
    Median absolute deviation of samples w.r.t. the observation s0.
    """
    mado = np.nanmedian(np.abs(samples - s0), axis=0)
    return mado


def mean_absolute_deviation_to_observation(samples, s0, **kwargs):
    """
    Mean absolute deviation of samples w.r.t. the observation s0.
    """
    mado = np.nanmean(np.abs(samples - s0), axis=0)
    return mado


def combined_median_absolute_deviation(samples, s0, **kwargs):
    """
    Compute the sum of the median absolute deviations to the
    median of the samples and to the observed value.
    """
    mad = median_absolute_deviation(samples)
    mado = median_absolute_deviation_to_observation(samples, s0)
    cmad = mad + mado
    return cmad


def combined_mean_absolute_deviation(samples, s0, **kwargs):
    """
    Compute the sum of the mean absolute deviations to the
    mean of the samples and to the observed value.
    """
    mad = mean_absolute_deviation(samples)
    mado = mean_absolute_deviation_to_observation(samples, s0)
    cmad = mad + mado
    return cmad


def standard_deviation_to_observation(samples, s0, **kwargs):
    """
    Standard deviation of absolute deviations of the samples w.r.t.
    the observation s0.
    """
    stdo = np.nanstd(np.abs(samples - s0), axis=0)
    return stdo


def span(samples, **kwargs):
    """
    Compute the difference of largest and smallest samples point.
    """
    return np.nanmax(samples, axis=0) - np.nanmin(samples, axis=0)


def mean(samples, **kwargs):
    """
    Compute the mean.
    """
    return np.nanmean(samples, axis=0)


def median(samples, **kwargs):
    """
    Compute the median.
    """
    return np.nanmedian(samples, axis=0)
