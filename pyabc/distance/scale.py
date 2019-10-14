"""
Various schemes to compute scales for the AdaptivePNormDistance and the
AdaptiveAggregatedDistance.

All of the functions take as argument a `data` list of values, and some in
addition other arguments, in particular a float `x_0` for the observed
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

For AdaptiveAggregatedDistance, for which the data are not summary statistic
values, but distance values, instead use either of
* span
* mean
* median
"""


import numpy as np


def median_absolute_deviation(data, **kwargs):
    """
    Calculate the sample `median absolute deviation (MAD)
    <https://en.wikipedia.org/wiki/Median_absolute_deviation/>`_
    from the median, defined as
    median(abs(data - median(data)).
    """
    data = np.array(data)
    mad = np.median(np.abs(data - np.median(data)))
    return mad


def mean_absolute_deviation(data, **kwargs):
    """
    Calculate the mean absolute deviation from the mean.
    """
    data = np.array(data)
    mad = np.mean(np.abs(data - np.mean(data)))
    return mad


def standard_deviation(data, **kwargs):
    """
    Calculate the sample `standard deviation (SD)
    <https://en.wikipedia.org/wiki/Standard_deviation/>`_.
    """
    std = np.std(data)
    return std


def bias(data, x_0, **kwargs):
    """
    Bias of sample to observed value.
    """
    bias = np.abs(np.mean(data) - x_0)
    return bias


def root_mean_square_deviation(data, x_0, **kwargs):
    """
    Square root of the mean squared error, i.e.
    of the bias squared plus the variance.
    """
    bs = bias(data, x_0)
    std = standard_deviation(data)
    mse = bs**2 + std**2
    rmse = np.sqrt(mse)
    return rmse


def median_absolute_deviation_to_observation(data, x_0, **kwargs):
    """
    Median absolute deviation of data w.r.t. the observation x_0.
    """
    data = np.array(data)
    mado = np.median(np.abs(data - x_0))
    return mado


def mean_absolute_deviation_to_observation(data, x_0, **kwargs):
    """
    Mean absolute deviation of data w.r.t. the observation x_0.
    """
    data = np.array(data)
    mado = np.mean(np.abs(data - x_0))
    return mado


def combined_median_absolute_deviation(data, x_0, **kwargs):
    """
    Compute the sum of the median absolute deviations to the
    median of the samples and to the observed value.
    """
    mad = median_absolute_deviation(data)
    mado = median_absolute_deviation_to_observation(data, x_0)
    cmad = mad + mado
    return cmad


def combined_mean_absolute_deviation(data, x_0, **kwargs):
    """
    Compute the sum of the mean absolute deviations to the
    mean of the samples and to the observed value.
    """
    mad = mean_absolute_deviation(data)
    mado = mean_absolute_deviation_to_observation(data, x_0)
    cmad = mad + mado
    return cmad


def standard_deviation_to_observation(data, x_0, **kwargs):
    """
    Standard deviation of absolute deviations of the data w.r.t.
    the observation x_0.
    """
    data = np.array(data)
    stdo = np.std(np.abs(data - x_0))
    return stdo


def span(data, **kwargs):
    """
    Compute the difference of largest and smallest data point.
    """
    return max(data) - min(data)


def mean(data, **kwargs):
    """
    Compute the mean.
    """
    return np.mean(data)


def median(data, **kwargs):
    """
    Compute the median.
    """
    return np.median(data)
