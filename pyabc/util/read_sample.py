"""Read sample to array."""

import numpy as np
from typing import Tuple

from ..population import Sample

from .par_trafo import ParTrafoBase


def _only_finites(*args):
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
    par_trafo: ParTrafoBase,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read in sample.

    Parameters
    ----------
    sample: Calibration or last generation's sample.
    sumstat: Up-chain summary statistic, already fitted.
    all_particles: Whether to use all particles or only accepted ones.
    par_trafo: Parameter transformation to apply.

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
    n_sumstat = len(sumstat(particles[0].sum_stat).flatten())
    n_par = len(par_trafo(particles[0].parameter))

    # prepare matrices
    sumstats = np.empty((n_sample, n_sumstat))
    parameters = np.empty((n_sample, n_par))
    weights = np.empty((n_sample, 1))

    # fill by iteration over all particles
    for i_particle, particle in enumerate(particles):
        sumstats[i_particle, :] = sumstat(particle.sum_stat).flatten()
        parameters[i_particle, :] = par_trafo(particle.parameter)
        weights[i_particle] = particle.weight

    # remove samples where an entry is not finite
    sumstats, parameters, weights = _only_finites(
        sumstats, parameters, weights,
    )

    return sumstats, parameters, weights
