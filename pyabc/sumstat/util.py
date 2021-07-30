"Utility functions."

import numpy as np
import pandas as pd
from numbers import Number
from typing import List, Tuple, Union

from ..population import Sample


def dict2arr(dct: Union[dict, np.ndarray], keys: List) -> np.ndarray:
    """Convert dictionary to 1d array, in specified key order.

    Parameters
    ----------
    dct: If dict-similar, values of all keys are extracted into a 1d array.
         Entries can be data frames, ndarrays, or single numbers.
    keys: Keys of interest, also defines the order.

    Returns
    -------
    arr: 1d array of all concatenated values.
    """
    if isinstance(dct, np.ndarray):
        return dct

    arr = []
    for key in keys:
        val = dct[key]
        if isinstance(val, (pd.DataFrame, pd.Series)):
            arr.append(val.to_numpy().flatten())
        elif isinstance(val, np.ndarray):
            arr.append(val.flatten())
        elif isinstance(val, Number):
            arr.append([val])
        else:
            raise TypeError(
                f"Cannot parse variable {key}={val} of type {type(val)} "
                "to numeric.")

    # for efficiency, directly parse single entries
    if len(arr) == 1:
        return np.asarray(arr[0])
    # flatten
    arr = [val for sub_arr in arr for val in sub_arr]
    return np.asarray(arr)


def dict2arrlabels(dct: dict, keys: List) -> List[str]:
    """Get label array consistent with the output of `dict2arr`.

    Can be called e.g. once on the observed data and used for logging.

    Parameters
    ----------
    dct: Model output or observed data.
    keys: Keys of interest, also defines the order.

    Returns
    -------
    labels: List of labels consistent with the output of `dict2arr`.
    """
    labels = []
    for key in keys:
        val = dct[key]
        if isinstance(val, (pd.DataFrame, pd.Series)):
            # default flattening mode is 'C', i.e. row-major, i.e. row-by-row
            for row in range(len(val.index)):
                for col in val.columns:
                    labels.append(f"{key}:{col}:{row}")
        elif isinstance(val, np.ndarray):
            # array can have any dimension, thus just flat indices
            for ix in range(val.size):
                labels.append(f"{key}:{ix}")
        elif isinstance(val, Number):
            labels.append(key)
        else:
            raise TypeError(
                f"Cannot parse variable {key}={val} of type {type(val)} "
                "to numeric.")
    return labels


def io_dict2arr(fun):
    """Wrapper parsing inputs dicts to ndarrays.

    Assumes the array is the first argument, and `self` holds a `keys`
    variable.
    """
    def wrapped_fun(self, data: Union[dict, np.ndarray], *args, **kwargs):
        # convert input to array
        data = dict2arr(data, self.x_keys)
        # call the actual function
        ret: np.ndarray = fun(self, data, *args, **kwargs)
        # flatten output
        return ret.flatten()
    return wrapped_fun


def only_finites(*args):
    """Remove samples (rows) where any entry is not finite.

    Parameters
    ----------
    A collection of np.ndarray objects, each of shape (n_sample, n_x) or
    (n_sample,).

    Returns
    -------
    The objects excluding rows where any entry in any object is not finite.
    """
    # create array of rows to keep
    keep = np.ones((args[0].shape[0],), dtype=bool)
    # check each argument whether a row has non-finite entries
    for arg in args:
        if arg.ndim == 1:
            keep = np.logical_and(keep, np.isfinite(arg))
        else:
            keep = np.logical_and(keep, np.all(np.isfinite(arg), axis=1))

    # reduce arrays
    args = [arg[keep] for arg in args]

    return args


def read_sample(
    sample: Sample,
    sumstat,
    all_particles: bool,
    par_keys: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read in sample.

    Parameters
    ----------
    sample: Calibration or last generation's sample.
    sumstat: Up-chain summary statistic, already fitted.
    all_particles: Whether to use all particles or only accepted ones.
    par_keys: Parameter keys, for correct order.

    Returns
    -------
    sumstats, parameters, weights: Arrays of shape (n_sample, n_out).
    """
    if all_particles:
        particles = sample.all_particles
    else:
        particles = sample.accepted_particles

    # dimensions of sample, summary statistics, and parameters
    n_sample = len(particles)
    n_sumstat = len(sumstat(particles[0].sum_stat))
    n_par = len(particles[0].parameter)

    # prepare matrices
    sumstats = np.empty((n_sample, n_sumstat))
    parameters = np.empty((n_sample, n_par))
    weights = np.empty((n_sample, 1))

    # fill by iteration over all particles
    for i_particle, particle in enumerate(particles):
        sumstats[i_particle, :] = sumstat(particle.sum_stat)
        parameters[i_particle, :] = dict2arr(particle.parameter, keys=par_keys)
        weights[i_particle] = particle.weight

    # remove samples where an entry is not finite
    sumstats, parameters, weights = only_finites(sumstats, parameters, weights)

    return sumstats, parameters, weights
