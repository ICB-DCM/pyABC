"""Inference utilities."""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Callable
import uuid

from pyabc.acceptor import Acceptor
from pyabc.distance import Distance
from pyabc.epsilon import Epsilon
from pyabc.model import Model
from pyabc.random_variables import RV, Distribution
from pyabc.transition import Transition, ModelPerturbationKernel
from pyabc.random_choice import fast_random_choice
from pyabc.parameters import Parameter
from pyabc.population import Particle

logger = logging.getLogger(__name__)


class AnalysisVars:
    """Contract object class for passing analysis variables.

    Used e.g. to create new sampling tasks or check early stopping.
    """

    def __init__(
        self,
        model_prior: RV,
        parameter_priors: List[Distribution],
        model_perturbation_kernel: ModelPerturbationKernel,
        transitions: List[Transition],
        nr_samples_per_parameter: int,
        models: List[Model],
        summary_statistics: Callable,
        x_0: dict,
        distance_function: Distance,
        eps: Epsilon,
        acceptor: Acceptor,
        min_acceptance_rate: float,
        min_eps: float,
        stop_if_single_model_alive: bool,
        max_t: int,
        max_total_nr_simulations: int,
        prev_total_nr_simulations: int,
        max_walltime: timedelta,
        init_walltime: datetime,
    ):
        self.model_prior = model_prior
        self.parameter_priors = parameter_priors
        self.model_perturbation_kernel = model_perturbation_kernel
        self.transitions = transitions
        self.nr_samples_per_parameter = nr_samples_per_parameter
        self.models = models
        self.summary_statistics = summary_statistics
        self.x_0 = x_0
        self.distance_function = distance_function
        self.eps = eps
        self.acceptor = acceptor
        self.min_acceptance_rate = min_acceptance_rate
        self.min_eps = min_eps
        self.stop_if_single_model_alive = stop_if_single_model_alive
        self.max_t = max_t
        self.max_total_nr_simulations = max_total_nr_simulations
        self.prev_total_nr_simulations = prev_total_nr_simulations
        self.max_walltime = max_walltime
        self.init_walltime = init_walltime


def create_simulate_from_prior_function(
        t: int, model_prior: RV, parameter_priors: List[Distribution],
        models: List[Model], summary_statistics: Callable,
) -> Callable:
    """Create a function that simulates from the prior.

    Similar to _create_simulate_function, apart here we sample from the
    prior and accept all.

    Parameters
    ----------
    t: The time to create the simulation function for.
    model_prior: The model prior.
    parameter_priors: The parameter priors.
    models: List of all models.
    summary_statistics:
        Function to compute summary statistics from model output.

    Returns
    -------
    simulate_one:
        A function that returns a sampled particle.
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
        t: int, m: np.ndarray, p: np.ndarray,
        model_prior: RV, parameter_priors: List[Distribution],
        model_perturbation_kernel: ModelPerturbationKernel,
        transitions: List[Transition]):
    """Sample a parameter for a model.

    Parameters
    ----------
    t: Population index to generate for.
    m: Indices of alive models.
    p: Probabilities of alive models.
    model_prior: The model prior.
    parameter_priors: The parameter priors.
    model_perturbation_kernel: The model perturbation kernel.
    transitions: The transitions, one per model.

    Returns
    -------
    (m_ss, theta_ss): Model, parameter.
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
            # only one model
            m_ss = m[0]
        theta_ss = Parameter(**transitions[m_ss].rvs().to_dict())

        # check if positive under prior
        if (model_prior.pmf(m_ss)
                * parameter_priors[m_ss].pdf(theta_ss) > 0):
            return m_ss, theta_ss

        # unhealthy sampling detection
        n_sample += 1
        if n_sample == n_sample_soft_limit:
            logger.warning(
                "Unusually many (model, parameter) samples have prior "
                "density zero. The transition might be inappropriate.")


def evaluate_proposal(
        m_ss: int, theta_ss: Parameter, t: int,
        nr_samples_per_parameter: int, models: List[Model],
        summary_statistics: Callable,
        distance_function: Distance, eps: Epsilon, acceptor: Acceptor,
        x_0: dict, weight_function: Callable) -> Particle:
    """Evaluate a proposed parameter.

    Parameters
    ----------
    m_ss, theta_ss: The proposed (model, parameter) sample.
    t: The current time.
    nr_samples_per_parameter: Number of samples per parameter.
    models: List of all models.
    summary_statistics:
        Function to compute summary statistics from model output.
    distance_function: The distance function.
    eps: The epsilon threshold.
    acceptor: The acceptor.
    x_0: The observed summary statistics.
    weight_function: Function by which to reweight the sample.

    Returns
    -------
    particle: A particle containing all information.

    Data for the given parameters theta_ss are simulated, summary statistics
    computed and evaluated.
    """
    accepted_sum_stats = []
    accepted_distances = []
    rejected_sum_stats = []
    rejected_distances = []
    accepted_weights = []

    # perform nr_samples_per_parameter simulations and check acceptance
    for _ in range(nr_samples_per_parameter):
        # simulate, compute distance, check acceptance
        model_result = models[m_ss].accept(
            t,
            theta_ss,
            summary_statistics,
            distance_function,
            eps,
            acceptor,
            x_0)
        # check whether to append to accepted particles
        if model_result.accepted:
            accepted_sum_stats.append(model_result.sum_stats)
            accepted_distances.append(model_result.distance)
            accepted_weights.append(model_result.weight)
        else:
            rejected_sum_stats.append(model_result.sum_stats)
            rejected_distances.append(model_result.distance)

    # check whether any simulation got accepted
    accepted = len(accepted_sum_stats) > 0

    # compute acceptance weight
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


def create_prior_pdf(
        model_prior: RV, parameter_priors: List[Distribution]) -> Callable:
    """Create a function that calculates a sample's prior density.

    Parameters
    ----------
    model_prior: The model prior.
    parameter_priors: The parameter priors, one for each model.

    Returns
    -------
    prior_pdf: The prior density function.
    """
    def prior_pdf(m_ss, theta_ss):
        prior_pd = (model_prior.pmf(m_ss)
                    * parameter_priors[m_ss].pdf(theta_ss))
        return prior_pd

    return prior_pdf


def create_transition_pdf(
        transitions: List[Transition],
        model_probabilities: pd.DataFrame,
        model_perturbation_kernel: ModelPerturbationKernel) -> Callable:
    """Create the transition probability density function for time `t`.

    Parameters
    ----------
    transitions: The list of parameter transition functions.
    model_probabilities: The last generation's model probabilities.
    model_perturbation_kernel: The kernel perturbing the models.

    Returns
    -------
    transition_pdf: The transition density function.
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
        nr_samples_per_parameter: int,
        prior_pdf: Callable,
        transition_pdf: Callable) -> Callable:
    """Create a function that calculates a sample's importance weight.
    The weight is the prior divided by the transition density and the
    acceptance step weight.

    Parameters
    ----------
    nr_samples_per_parameter: Number of samples per parameter.
    prior_pdf: The prior density.
    transition_pdf: The transition density.

    Returns
    -------
    weight_function: The importance sample weight function.
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
        t: int,
        model_probabilities: pd.DataFrame,
        model_perturbation_kernel: ModelPerturbationKernel,
        transitions: List[Transition],
        model_prior: RV,
        parameter_priors: List[Distribution],
        nr_samples_per_parameter: int,
        models: List[Model],
        summary_statistics: Callable,
        x_0: dict,
        distance_function: Distance,
        eps: Epsilon,
        acceptor: Acceptor,
        evaluate: bool = True,
) -> Callable:
    """
    Create a simulation function which performs the sampling of parameters,
    simulation of data and acceptance checking, and which is then passed
    to the sampler.

    Parameters
    ----------
    t: The time index to simulate for.
    model_probabilities: The last generation's model probabilities.
    model_perturbation_kernel: The model perturbation kernel.
    transitions: The parameter transition kernels.
    model_prior: The model prior.
    parameter_priors: The parameter priors.
    nr_samples_per_parameter: Number of samples per parameter.
    models: List of all models.
    summary_statistics:
        Function to compute summary statistics from model output.
    x_0: The observed summary statistics.
    distance_function: The distance function.
    eps: The epsilon threshold.
    acceptor: The acceptor.
    evaluate:
        Whether to actually evaluate the sample. Should be True except for
        certain preliminary settings.

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

    # create prior and transition densities for weight function
    prior_pdf = create_prior_pdf(
        model_prior=model_prior, parameter_priors=parameter_priors)
    if t == 0:
        transition_pdf = prior_pdf
    else:
        transition_pdf = create_transition_pdf(
            transitions=transitions,
            model_probabilities=model_probabilities,
            model_perturbation_kernel=model_perturbation_kernel)

    # create weight function
    weight_function = create_weight_function(
        nr_samples_per_parameter=nr_samples_per_parameter,
        prior_pdf=prior_pdf, transition_pdf=transition_pdf)

    # simulation function
    def simulate_one():
        parameter = generate_valid_proposal(
            t=t, m=m, p=p,
            model_prior=model_prior, parameter_priors=parameter_priors,
            model_perturbation_kernel=model_perturbation_kernel,
            transitions=transitions)
        if evaluate:
            particle = evaluate_proposal(
                *parameter, t=t,
                nr_samples_per_parameter=nr_samples_per_parameter,
                models=models, summary_statistics=summary_statistics,
                distance_function=distance_function, eps=eps,
                acceptor=acceptor,
                x_0=x_0, weight_function=weight_function)
        else:
            particle = only_simulate_data_for_proposal(
                *parameter, t=t,
                nr_samples_per_parameter=nr_samples_per_parameter,
                models=models, summary_statistics=summary_statistics,
                weight_function=weight_function)
        return particle

    return simulate_one


def only_simulate_data_for_proposal(
        m_ss: int, theta_ss: Parameter, t: int,
        nr_samples_per_parameter: int, models: List[Model],
        summary_statistics: Callable,
        weight_function: Callable) -> Particle:
    """Simulate data for parameters.

    Similar to `evaluate_proposal`, however here for the passed parameters
    only data are simulated, but no distances calculated or acceptance
    checked. That needs to be done post-hoc then, not checked here."""
    # for the results
    accepted_sum_stats = []
    # distance and weight are just dummies here, they need to be recomputed
    #  later again
    accepted_distances = []
    accepted_weights = []

    # perform nr_samples_per_parameter simulations
    for _ in range(nr_samples_per_parameter):
        # simulate
        model_result = models[m_ss].summary_statistics(
            t, theta_ss, summary_statistics)
        accepted_sum_stats.append(model_result.sum_stats)
        # fill in dummies for distance and weight
        accepted_distances.append(np.inf)
        accepted_weights.append(1.)

    # needs to be accepted in order to be forwarded by the sampler, and so
    #  as a single particle
    accepted = True

    # compute acceptance weight
    # TODO later replacement only works with nr_samples_per_parameter == 1
    weight = weight_function(
        accepted_distances, m_ss, theta_ss, accepted_weights)

    return Particle(
        m=m_ss,
        parameter=theta_ss,
        weight=weight,
        accepted_sum_stats=accepted_sum_stats,
        accepted_distances=accepted_distances,
        accepted=accepted,
        preliminary=True)


def evaluate_preliminary_particle(
        particle: Particle, t, ana_vars: AnalysisVars) -> Particle:
    """Evaluate a preliminary particle.
    I.e. compute distance and check acceptance.

    Returns
    -------
    evaluated_particle: The evaluated particle
    """
    if not particle.preliminary:
        raise AssertionError("Particle is not preliminary")

    # for results
    accepted_sum_stats = []
    accepted_distances = []
    accepted_weights = []
    rejected_sum_stats = []
    rejected_distances = []

    for sum_stat in particle.accepted_sum_stats:
        acc_res = ana_vars.acceptor(
            distance_function=ana_vars.distance_function,
            eps=ana_vars.eps,
            x=sum_stat,
            x_0=ana_vars.x_0,
            t=t,
            par=particle.parameter)

        if acc_res.accept:
            accepted_sum_stats.append(sum_stat)
            accepted_distances.append(acc_res.distance)
            # the acceptance weight
            accepted_weights.append(acc_res.weight)
        else:
            rejected_sum_stats.append(sum_stat)
            rejected_distances.append(acc_res.distance)

    # reconstruct weighting function from `weight_function`
    sampling_weight = particle.weight
    fr_accepted_for_par = \
        len(accepted_sum_stats) / ana_vars.nr_samples_per_parameter
    # the weight is the sampling weight times the acceptance weight(s)
    weight = sampling_weight * np.prod(accepted_weights) * \
        fr_accepted_for_par

    # return the evaluated particle
    return Particle(
        m=particle.m,
        parameter=particle.parameter,
        weight=weight,
        accepted_sum_stats=accepted_sum_stats,
        accepted_distances=accepted_distances,
        rejected_sum_stats=rejected_sum_stats,
        rejected_distances=rejected_distances,
        accepted=len(accepted_distances) > 0,
    )


def termination_criteria_fulfilled(
        current_eps: float, min_eps: float,
        stop_if_single_model_alive: bool, nr_of_models_alive: int,
        acceptance_rate: float, min_acceptance_rate: float,
        total_nr_simulations: int, max_total_nr_simulations: int,
        walltime: timedelta, max_walltime: timedelta,
        t: int, max_t: int) -> bool:
    """Check termination criteria.

    Parameters
    ----------
    current_eps: The last generation's epsilon value.
    min_eps: The minimum allowed epsilon value.
    stop_if_single_model_alive: Whether to stop with a single model left.
    nr_of_models_alive: The number of models alive in the last generation.
    acceptance_rate: The last generation's acceptance rate.
    min_acceptance_rate: The minimum acceptance rate.
    total_nr_simulations: The total number of simulations so far.
    max_total_nr_simulations: Bound on the total number of simulations.
    walltime: Walltime passed since start of the analysis.
    max_walltime: Maximum allowed walltime.
    t: The last generation's time index.
    max_t: The maximum allowed time index.

    Returns
    -------
    True if any criterion is met, otherwise False.
    """
    if t >= max_t:
        logger.info("Stopping: maximum number of generations.")
        return True
    if current_eps <= min_eps:
        logger.info("Stopping: minimum epsilon.")
        return True
    elif stop_if_single_model_alive and nr_of_models_alive <= 1:
        logger.info("Stopping: single model alive.")
        return True
    elif acceptance_rate < min_acceptance_rate:
        logger.info("Stopping: minimum acceptance rate.")
        return True
    elif total_nr_simulations >= max_total_nr_simulations:
        logger.info("Stopping: total simulations budget.")
        return True
    elif max_walltime is not None and walltime > max_walltime:
        logger.info("Stopping: maximum walltime.")
        return True
    return False


def create_analysis_id():
    """Create a universally unique id for a given analysis.
    Used by the inference routine to uniquely associated results with analyses.
    """
    return str(uuid.uuid4())
