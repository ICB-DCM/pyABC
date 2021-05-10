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

logger = logging.getLogger("ABC")


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
        model_prior: RV, parameter_priors: List[Distribution],
        models: List[Model], summary_statistics: Callable,
) -> Callable:
    """Create a function that simulates from the prior.

    Similar to _create_simulate_function, apart here we sample from the
    prior and accept all.

    Parameters
    ----------
    model_prior: The model prior.
    parameter_priors: The parameter priors.
    models: List of all models.
    summary_statistics: Computes summary statistics from model output.

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
            0, theta, summary_statistics)
        # sampled from prior, so all have uniform weight
        weight = 1.0
        # distance will be computed after initialization of the
        #  distance function
        distance = np.inf
        # all are happy and accepted
        accepted = True

        return Particle(
            m=m,
            parameter=theta,
            weight=weight,
            sum_stat=model_result.sum_stat,
            distance=distance,
            accepted=accepted,
            proposal_id=0,
            preliminary=False)

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
        theta_ss = transitions[m_ss].rvs()

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
        models: List[Model],
        summary_statistics: Callable,
        distance_function: Distance, eps: Epsilon, acceptor: Acceptor,
        x_0: dict, weight_function: Callable, proposal_id: int,
) -> Particle:
    """Evaluate a proposed parameter.

    Parameters
    ----------
    m_ss, theta_ss: The proposed (model, parameter) sample.
    t: The current time.
    models: List of all models.
    summary_statistics:
        Function to compute summary statistics from model output.
    distance_function: The distance function.
    eps: The epsilon threshold.
    acceptor: The acceptor.
    x_0: The observed summary statistics.
    weight_function: Function by which to reweight the sample.
    proposal_id: Id of the transition kernel.

    Returns
    -------
    particle: A particle containing all information.

    Data for the given parameters theta_ss are simulated, summary statistics
    computed and evaluated.
    """
    # simulate, compute distance, check acceptance
    model_result = models[m_ss].accept(
        t,
        theta_ss,
        summary_statistics,
        distance_function,
        eps,
        acceptor,
        x_0)

    # compute acceptance weight
    if model_result.accepted:
        weight = weight_function(m_ss, theta_ss, model_result.weight)
    else:
        weight = 0

    return Particle(
        m=m_ss,
        parameter=theta_ss,
        weight=weight,
        sum_stat=model_result.sum_stat,
        distance=model_result.distance,
        accepted=model_result.accepted,
        preliminary=False,
        proposal_id=proposal_id)


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
        particle_factor = transitions[m_ss].pdf(theta_ss)

        transition_pd = model_factor * particle_factor

        if transition_pd == 0:
            logger.debug("Transition density is zero!")
        return transition_pd

    return transition_pdf


def create_weight_function(
        prior_pdf: Callable,
        transition_pdf: Callable,
) -> Callable:
    """Create a function that calculates a sample's importance weight.
    The weight is the prior divided by the transition density and the
    acceptance step weight.

    Parameters
    ----------
    prior_pdf: The prior density.
    transition_pdf: The transition density.

    Returns
    -------
    weight_function: The importance sample weight function.
    """
    def weight_function(m_ss, theta_ss, acceptance_weight: float):
        """Calculate total weight, from sampling and acceptance weight.

        Parameters
        ----------
        m_ss: The model sample.
        theta_ss: The parameter sample.
        acceptance_weight: The acceptance weight sample. In most cases 1.

        Returns
        -------
        weight: The total weight.
        """
        # prior and transition density (can be equal)
        prior_pd = prior_pdf(m_ss, theta_ss)
        transition_pd = transition_pdf(m_ss, theta_ss)
        # calculate weight
        weight = acceptance_weight * prior_pd / transition_pd
        return weight

    return weight_function


def create_simulate_function(
        t: int,
        model_probabilities: pd.DataFrame,
        model_perturbation_kernel: ModelPerturbationKernel,
        transitions: List[Transition],
        model_prior: RV,
        parameter_priors: List[Distribution],
        models: List[Model],
        summary_statistics: Callable,
        x_0: dict,
        distance_function: Distance,
        eps: Epsilon,
        acceptor: Acceptor,
        evaluate: bool = True,
        proposal_id: int = 0,
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
    proposal_id:
        Identifier for the proposal distribution.

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
                models=models, summary_statistics=summary_statistics,
                distance_function=distance_function, eps=eps,
                acceptor=acceptor,
                x_0=x_0, weight_function=weight_function,
                proposal_id=proposal_id)
        else:
            particle = only_simulate_data_for_proposal(
                *parameter, t=t,
                models=models, summary_statistics=summary_statistics,
                weight_function=weight_function, proposal_id=proposal_id)
        return particle

    return simulate_one


def only_simulate_data_for_proposal(
        m_ss: int, theta_ss: Parameter, t: int,
        models: List[Model],
        summary_statistics: Callable,
        weight_function: Callable,
        proposal_id: int,
) -> Particle:
    """Simulate data for parameters.

    Similar to `evaluate_proposal`, however here for the passed parameters
    only data are simulated, but no distances calculated or acceptance
    checked. That needs to be done post-hoc then, not checked here."""

    # simulate
    model_result = models[m_ss].summary_statistics(
        t, theta_ss, summary_statistics)

    # dummies for distance and weight, need to be recomputed later
    distance = np.inf
    acceptance_weight = 1.

    # needs to be accepted in order to be forwarded by the sampler, and so
    #  as a single particle
    accepted = True

    # compute weight
    weight = weight_function(m_ss, theta_ss, acceptance_weight)

    return Particle(
        m=m_ss,
        parameter=theta_ss,
        weight=weight,
        sum_stat=model_result.sum_stat,
        distance=distance,
        accepted=accepted,
        preliminary=True,
        proposal_id=proposal_id,
    )


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

    acc_res = ana_vars.acceptor(
        distance_function=ana_vars.distance_function,
        eps=ana_vars.eps,
        x=particle.sum_stat,
        x_0=ana_vars.x_0,
        t=t,
        par=particle.parameter)

    # reconstruct weighting function from `weight_function`
    sampling_weight = particle.weight
    # the weight is the sampling weight times the acceptance weight(s)
    if acc_res.accept:
        weight = sampling_weight * acc_res.weight
    else:
        weight = 0

    # return the evaluated particle
    return Particle(
        m=particle.m,
        parameter=particle.parameter,
        weight=weight,
        sum_stat=particle.sum_stat,
        distance=acc_res.distance,
        accepted=acc_res.accept,
        preliminary=False,
        proposal_id=particle.proposal_id,
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
        logger.info("Stop: Maximum number of generations.")
        return True
    if current_eps <= min_eps:
        logger.info("Stop: Minimum epsilon.")
        return True
    elif stop_if_single_model_alive and nr_of_models_alive <= 1:
        logger.info("Stop: Single model alive.")
        return True
    elif acceptance_rate < min_acceptance_rate:
        logger.info("Stop: Minimum acceptance rate.")
        return True
    elif total_nr_simulations >= max_total_nr_simulations:
        logger.info("Stop: Total simulations budget.")
        return True
    elif max_walltime is not None and walltime > max_walltime:
        logger.info("Stop: Maximum walltime.")
        return True
    return False


def create_analysis_id():
    """Create a universally unique id for a given analysis.
    Used by the inference routine to uniquely associated results with analyses.
    """
    return str(uuid.uuid4())
