"""
Various schemes to compute scales for the
(Adaptive)PNormDistance.

For usage, the following are recommended:
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
"""


import numpy as np


def median_absolute_deviation(**kwargs):
    """
    Calculate the sample `median absolute deviation (MAD)
    <https://en.wikipedia.org/wiki/Median_absolute_deviation/>`_
    from the median, defined as
    median(abs(data - median(data)).
    """
    data = np.asarray(kwargs['data'])
    mad = np.median(np.abs(data - np.median(data)))
    return mad


def mean_absolute_deviation(**kwargs):
    """
    Calculate the mean absolute deviation from the mean.
    """
    data = np.asarray(kwargs['data'])
    mad = np.mean(np.abs(data - np.mean(data)))
    return mad


def standard_deviation(**kwargs):
    """
    Calculate the sample `standard deviation (SD)
    <https://en.wikipedia.org/wiki/Standard_deviation/>`_.
    """
    data = np.asarray(kwargs['data'])
    std = np.std(data)
    return std


def bias(**kwargs):
    """
    Bias of sample to observed value.
    """
    data = np.asarray(kwargs['data'])
    x_0 = kwargs['x_0']
    bias = np.abs(np.mean(data) - x_0)
    return bias


def root_mean_square_deviation(**kwargs):
    """
    Square root of the mean squared error, i.e.
    of the bias squared plus the variance.
    """
    bs = bias(**kwargs)
    std = standard_deviation(**kwargs)
    mse = bs**2 + std**2
    rmse = np.sqrt(mse)
    return rmse


def median_absolute_deviation_to_observation(**kwargs):
    """
    Median absolute deviation of data w.r.t. the observation x_0.
    """
    data = np.asarray(kwargs['data'])
    x_0 = kwargs['x_0']
    mado = np.median(np.abs(data - x_0))
    return mado


def mean_absolute_deviation_to_observation(**kwargs):
    """
    Mean absolute deviation of data w.r.t. the observation x_0.
    """
    data = np.asarray(kwargs['data'])
    x_0 = kwargs['x_0']
    mado = np.mean(np.abs(data - x_0))
    return mado


def combined_median_absolute_deviation(**kwargs):
    """
    Compute the sum of the median absolute deviations to the
    median of the samples and to the observed value.
    """
    mad = median_absolute_deviation(**kwargs)
    mado = median_absolute_deviation_to_observation(**kwargs)
    cmad = mad + mado
    return cmad


def combined_mean_absolute_deviation(**kwargs):
    """
    Compute the sum of the mean absolute deviations to the
    mean of the samples and to the observed value.
    """
    mad = mean_absolute_deviation(**kwargs)
    mado = mean_absolute_deviation_to_observation(**kwargs)
    cmad = mad + mado
    return cmad


def standard_deviation_to_observation(**kwargs):
    """
    Standard deviation of absolute deviations of the data w.r.t.
    the observation x_0.
    """
    data = np.asarray(kwargs['data'])
    x_0 = kwargs['x_0']
    stdo = np.std(np.abs(data - x_0))
    return stdo
