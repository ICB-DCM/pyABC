from typing import Union

import scipy as sp

from .random_variables import NonEmptyMultivariateMultiTypeNormalDistribution, EmptyMultivariateMultiTypeNormalDistribution, \
    MultivariateMultiTypeNormalDistribution


def cov(particles: list) -> Union[
    NonEmptyMultivariateMultiTypeNormalDistribution, EmptyMultivariateMultiTypeNormalDistribution]:
    """
    Covariance from particles.

    Parameters
    ----------

    particles: list
        List of particles

    Returns
    -------

    cov: Union[NonEmptyMultivariateMultiTypeNormalDistribution, EmptyMultivariateMultiTypeNormalDistribution]
        The covariance representing distribution.
    """
    parameter_names = list(particles[0]['parameter'].keys())
    parameter_names = sorted(parameter_names)
    nr_parameters = len(parameter_names)
    nr_particles = len(particles)
    pars = sp.empty((nr_particles, nr_parameters))
    weights = sp.empty(nr_particles)
    parameter_types = [type(particles[0]['parameter'][k]) for k in parameter_names]
    for nr, particle in enumerate(particles):
        weights[nr] = particle['weight']
        for k, name in enumerate(parameter_names):
            pars[nr, k] = particle['parameter'][name]
    expectation = sum(w * x for w, x in zip(weights, pars))
    pars -= expectation
    cov = sum(w * sp.dot(x[None].T, x[None]) for w, x in zip(weights, pars))
    return MultivariateMultiTypeNormalDistribution(cov, parameter_names, parameter_types)