import numpy as np
import pandas as pd
import logging
from typing import Callable

from pyabc.pyabc_rand_choice import fast_random_choice
from pyabc.parameters import Parameter
from pyabc.population import Particle

logger = logging.getLogger(__name__)


def create_simulate_from_prior_function(
        t: int, model_prior, parameter_priors, models, summary_statistics):
    """Create a function that simulates from the prior.

    Similar to _create_simulate_function, apart here we sample from the
    prior and accept all.
    """
    # simulation function, simplifying some parts compared to later

    def simulate_one():
        # sample model
        m = int(model_prior.rvs())
        # sample parameter
        theta = parameter_priors[m].rvs()
        # simulate summary statistics
        model_result = models[m].summary_statistics(
            t, theta, summary_statistics)
        # sampled from prior, so all have uniform weight
        weight = 1.0
        # remember sum stat as accepted
        accepted_sum_stats = [model_result.sum_stats]
        # distance will be computed after initialization of the
        # distance function
        accepted_distances = [np.inf]
        # all are happy and accepted
        accepted = True

        return Particle(
            m=m,
            parameter=theta,
            weight=weight,
            accepted_sum_stats=accepted_sum_stats,
            accepted_distances=accepted_distances,
            rejected_sum_stats=[],
            rejected_distances=[],
            accepted=accepted)

    return simulate_one


def generate_valid_proposal(
        t, m, p,
        model_prior,
        parameter_priors,
        model_perturbation_kernel,
        transitions):
    """Sample a parameter for a model.

    Parameters
    ----------
    t: Population number
    m: Indices of alive models
    p: Probabilities of alive models

    Returns
    -------
    Model, parameter.
    """
    # first generation
    if t == 0:
        # sample from prior
        m_ss = int(model_prior.rvs())
        theta_ss = parameter_priors[m_ss].rvs()
        return m_ss, theta_ss

    # later generation
    # counter
    n_sample, n_sample_soft_limit = 0, 1000
    # sample until the prior density is positive
    while True:
        if len(m) > 1:
            index = fast_random_choice(p)
            m_s = m[index]
            m_ss = model_perturbation_kernel.rvs(m_s)
            # theta_s is None if the population m_ss has died out.
            # This can happen since the model_perturbation_kernel
            # can return a model nr which has died out.
            if m_ss not in m:
                continue
        else:
            m_ss = m[0]
        theta_ss = Parameter(**transitions[m_ss].rvs().to_dict())

        if (model_prior.pmf(m_ss)
                * parameter_priors[m_ss].pdf(theta_ss) > 0):
            return m_ss, theta_ss

        n_sample += 1
        if n_sample == n_sample_soft_limit:
            logger.warning(
                "Unusually many (model, parameter) samples have prior "
                "density zero. The transition might be inappropriate.")


def evaluate_proposal(
        m_ss, theta_ss,
        t,
        nr_samples_per_parameter,
        models,
        summary_statistics,
        distance_function,
        eps,
        acceptor,
        x_0,
        weight_function) -> Particle:
    """Evaluate a proposed parameter.

    Data for the given parameters theta_ss are simulated, summary statistics
    computed and evaluated.
    """

    # from here, theta_ss is valid according to the prior

    accepted_sum_stats = []
    accepted_distances = []
    rejected_sum_stats = []
    rejected_distances = []
    accepted_weights = []

    for _ in range(nr_samples_per_parameter):
        model_result = models[m_ss].accept(
            t,
            theta_ss,
            summary_statistics,
            distance_function,
            eps,
            acceptor,
            x_0)
        if model_result.accepted:
            accepted_sum_stats.append(model_result.sum_stats)
            accepted_distances.append(model_result.distance)
            accepted_weights.append(model_result.weight)
        else:
            rejected_sum_stats.append(model_result.sum_stats)
            rejected_distances.append(model_result.distance)

    accepted = len(accepted_sum_stats) > 0

    if accepted:
        weight = weight_function(
            accepted_distances, m_ss, theta_ss, accepted_weights)
    else:
        weight = 0

    return Particle(
        m=m_ss,
        parameter=theta_ss,
        weight=weight,
        accepted_sum_stats=accepted_sum_stats,
        accepted_distances=accepted_distances,
        rejected_sum_stats=rejected_sum_stats,
        rejected_distances=rejected_distances,
        accepted=accepted)


def create_prior_pdf(model_prior, parameter_priors) -> Callable:
    """Create a function that calculates a sample's prior density.

    Parameters
    ----------
    model_prior:
        The model prior.
    parameter_priors:
        The parameter priors, one for each model.
    """
    def prior_pdf(m_ss, theta_ss):
        prior_pd = (model_prior.pmf(m_ss)
                    * parameter_priors[m_ss].pdf(theta_ss))
        return prior_pd

    return prior_pdf


def create_transition_pdf(
        transitions, model_probabilities, model_perturbation_kernel):
    """Create transition probability density function for time `t`.

    Parameters
    ----------
    TODO
    """
    def transition_pdf(m_ss, theta_ss):
        model_factor = sum(
            row.p * model_perturbation_kernel.pmf(m_ss, m)
            for m, row in model_probabilities.iterrows())
        particle_factor = transitions[m_ss].pdf(
            pd.Series(dict(theta_ss)))

        transition_pd = model_factor * particle_factor

        if transition_pd == 0:
            logger.debug("Transition density is zero!")
        return transition_pd

    return transition_pdf


def create_weight_function(
        nr_samples_per_parameter: int, prior_pdf, transition_pdf):
    """Create a function that calculates a sample's weight at time `t`.
    The weight is the prior divided by the transition density and the
    acceptance setp weight.
    """
    def weight_function(
            distance_list, m_ss, theta_ss, acceptance_weights):
        # prior and transition density (can be equal)
        prior_pd = prior_pdf(m_ss, theta_ss)
        transition_pd = transition_pdf(m_ss, theta_ss)

        # account for stochastic acceptance
        # TODO This is only valid for single samples (see #54)
        acceptance_weight = np.prod(acceptance_weights)

        # account for multiple tries
        fr_accepted_for_par = \
            len(distance_list) / nr_samples_per_parameter

        # calculate weight
        weight = (
            prior_pd * acceptance_weight / transition_pd
            * fr_accepted_for_par)
        return weight

    return weight_function


def create_simulate_function(
        t: int, model_probabilities, model_perturbation_kernel,
        transitions, model_prior, parameter_priors,
        nr_samples_per_parameter,
        models, summary_statistics, distance_function, eps, acceptor, x_0,
):
    """
    Create a simulation function which performs the sampling of parameters,
    simulation of data and acceptance checking, and which is then passed
    to the sampler.

    Parameters
    ----------
    t: int
        Time index

    Returns
    -------
    simulate_one: callable
        Function that samples parameters, simulates data, and checks
        acceptance.

    .. note::
        For some of the samplers, the sampling function needs to be
        serialized in order to be transported to where the sampling
        happens. Therefore, the returned function should be light, and
        in particular not contain references to the ABCSMC class.
    """
    # cache model_probabilities to not query the database so often
    m = np.array(model_probabilities.index)
    p = np.array(model_probabilities.p)

    prior_pdf = create_prior_pdf(
        model_prior=model_prior, parameter_priors=parameter_priors)
    if t == 0:
        transition_pdf = prior_pdf
    else:
        transition_pdf = create_transition_pdf(
            transitions=transitions,
            model_probabilities=model_probabilities,
            model_perturbation_kernel=model_perturbation_kernel)

    weight_function = create_weight_function(
        nr_samples_per_parameter=nr_samples_per_parameter,
        prior_pdf=prior_pdf, transition_pdf=transition_pdf)

    # simulation function
    def simulate_one():
        parameter = generate_valid_proposal(
            t, m, p,
            model_prior,
            parameter_priors,
            model_perturbation_kernel,
            transitions)
        particle = evaluate_proposal(
            *parameter,
            t,
            nr_samples_per_parameter,
            models,
            summary_statistics,
            distance_function,
            eps,
            acceptor,
            x_0,
            weight_function)
        return particle

    return simulate_one
