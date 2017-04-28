"""
Parallel Approximate Bayesian Computation - Sequential Monte Carlo
==================================================================


"""

import datetime
import logging
from typing import List, Callable, TypeVar
import pandas as pd
import scipy as sp
from .sampler import MulticoreParticleParallelSampler
from .distance_functions import DistanceFunction  # noqa: F401
from .distance_functions import to_distance
from .epsilon import Epsilon, MedianEpsilon
from .model import Model
from .parameters import ValidParticle
from .transition import Transition, MultivariateNormalTransition
from .random_variables import RV, ModelPerturbationKernel, Distribution
from .storage import History
from .populationstrategy import PopulationStrategy
from .random import fast_random_choice
from typing import Union
from .model import SimpleModel
from .populationstrategy import ConstantPopulationStrategy
import copy


abclogger = logging.getLogger("ABC")

model_output = TypeVar("model_output")


def identity(x):
    return x


class ABCSMC:
    """
    Approximate Bayesian Computation - Sequential Monte Carlo (ABCSMC).

    This is an implementation of an ABCSMC algorithm similar to [#tonistumpf]_


    Parameters
    ----------

    models: List[Callable[[Parameter], model_output]]
       Calling ``models[m](par)`` returns the raw model output
       of model ``m`` with the corresponding parameters ``par``.
       This raw output is then passed to summary_statistics.
       calculated summary statistics. Per default, the model is
       assumed to already return the calculated summary statistcs.
       The default summary_statistics function is therefore
       just the identity.

       Each callable represents thus one single model.

    parameter_priors: List[Distribution]
        A list of prior distributions for the models' parameters.
        Each list entry is the prior distribution for the corresponding model.

    distance_function: DistanceFunction
        Measures the distance of the tentatively sampled particle to the
        measured data.

    population_specification: int, PopulationStrategy, optional
        Specify the size of the population.
        If ``population_specification`` is an ``int``, then the size is
        constant. Adaptive population sizes are also possible by passing a
        :class:`pyabc.populationstrategy.PopulationStrategy` object.
        The default is 100 particles per population.

    summary_statistics: Callable[[model_output], dict]
        A function which takes the raw model output as returned by
        any ot the models and calculates the corresponding summary
        statistics. Note that the default value is just the identity
        function. I.e. the model is assumed to already calculate
        the summary statistics. However, in a model selection setting
        it can make sense to have the model produce some kind or raw output
        and then take the same summary statistics function for all the models.

    model_prior: RV
        A random variable giving the prior weights of the model classes.
        If the prior is uniform over the model classes
        this is something like ``RV("randint", 0, len(models))``.

    model_perturbation_kernel: ModelPerturbationKernel
        Kernel which governs with which probability to switch the model
        for a given sample.

    transitions: List[Transition], Transition, optional
        A list of :class:`pyabc.transition.Transition` objects
        or a single :class:`pyabc.transition.Transition` in case
        of a single model. Defaults to multivariate normal transitions for
        every model.

    eps: Epsilon, optional
        Accepts any :class:`pyabc.epsilon.Epsilon` subclass.
        The default is the :class:`pyabc.epsilon.MediaEpsilon` which adapts
        automatically. The object passed here determines how the acceptance
        threshold scheduling is performed.

    sampler:
        In some cases, a mapper implementation will require initialization
        to run properly, e.g. database connection, grid setup, etc...
        The sampler is an object that encapsulates this information.
        The default sampler will simply call the callable mapper at the right
        place; a more involved sampler will help the mapper-function to
        distribute function calls accross a distributed infrastructure.



    .. [#tonistumpf] Toni, Tina, and Michael P. H. Stumpf.
                  “Simulation-Based Model Selection for Dynamical
                  Systems in Systems and Population Biology.”
                  Bioinformatics 26, no. 1 (January 1, 2010):
                  104–10. doi:10.1093/bioinformatics/btp619.
    """
    def __init__(self, models: List[Model],
                 parameter_priors: List[Distribution],
                 distance_function,
                 population_specification: Union[PopulationStrategy, int]
                 = 100,
                 summary_statistics: Callable[[model_output], dict] = identity,
                 model_prior: RV = None,
                 model_perturbation_kernel: ModelPerturbationKernel = None,
                 transitions: List[Transition] = None,
                 eps: Epsilon = None,
                 sampler=None):

        if not isinstance(models, list):
            models = [models]
        models = list(map(SimpleModel.assert_model, models))
        self.models = models

        if not isinstance(parameter_priors, list):
            parameter_priors = [parameter_priors]
        self.parameter_priors = parameter_priors

        # sanity checks
        assert len(self.models) == len(self.parameter_priors), \
            "Number models and number parameter priors have to agree"

        self.distance_function = to_distance(distance_function)

        self.summary_statistics = summary_statistics

        if model_prior is None:
            model_prior = RV("randint", 0, len(self.models))
        self.model_prior = model_prior

        if model_perturbation_kernel is None:
            model_perturbation_kernel = ModelPerturbationKernel(
                len(self.models), probability_to_stay=.7)
        self.model_perturbation_kernel = model_perturbation_kernel

        if transitions is None:
            transitions = [MultivariateNormalTransition() for _ in self.models]
        if not isinstance(transitions, list):
            transitions = [transitions]
        self.transitions = transitions  # type: List[Transition]

        if eps is None:
            eps = MedianEpsilon()
        self.eps = eps

        if isinstance(population_specification, int):
            population_specification = ConstantPopulationStrategy(
                population_specification)
        self.population_strategy = population_specification

        if sampler is None:
            self.sampler = MulticoreParticleParallelSampler()
        else:
            self.sampler = sampler

        self.stop_if_only_single_model_alive = False
        self.x_0 = None
        self.history = None  # type: History
        self._points_sampled_from_prior = None

    def __getstate__(self):
        state_red_dict = self.__dict__.copy()
        del state_red_dict['sampler']
        # print(state_red_dict)
        return state_red_dict

    def do_not_stop_when_only_single_model_alive(self):
        """
        Calling this method causes the ABCSMC to still continue if only
        a single model is still alive. This is useful if the interest lies in
        estimating the model parameter as compared to doing model selection.

        The default behavior is to stop when only a single model is alive.
        """
        self.stop_if_only_single_model_alive = False

    def set_data(self, observed_summary_statistics: dict,
                 abc_options: Union[dict, str],
                 ground_truth_model: int = -1,
                 ground_truth_parameter: dict = None):
        """
        Set the data to be fitted.

        Parameters
        ----------

        observed_summary_statistics : dict
               **This is the really important parameter here**. It is of the
               form ``{'statistic_1' : val_1, 'statistic_2': val_2, ... }``.

               The dictionary provided here represents the measured data.
               Particle during ABCSMC sampling are compared against the
               summary statistics provided here.

        abc_options: Union[dict, str]
            If a string, it has to be a valid SQLAlchemy database identifier.

            If a dict, has to contain the key "db_path" which has to be a valid
            SQLAlchemy database identifier. Can contain an arbitrary number of
            additional keys, only for recording purposes. Store arbitrary
            meta information in this dictionary. Can be used for really
            anything.

        ground_truth_model: int, optional
            This is only meta data stored to the database, but not actually
            used for the ABCSMC algorithm If you want to predict your ABCSMC
            procedure against synthetic samples, you can use
            this parameter to indicate the ground truth model number.
            This helps with futher analysis. If you use actually measured data
            (and don't know the ground truth) you can set this to anything.
            A value if ``-1`` is recommended.

        ground_truth_parameter: dict, optional
            Similar to ``ground_truth_model``, this is only for recording
            purposes, but not used in the ABCSMC algorithm.
            This stores the parameters of the ground truth model
            if it was syntheticallyobtained.
        """

        # initialize
        if isinstance(abc_options, str):
            abc_options = {"db_path": abc_options}

        self.x_0 = observed_summary_statistics
        model_names = [model.name for model in self.models]

        self.history = History(abc_options["db_path"])

        if ground_truth_parameter is None:
            ground_truth_parameter = {}

        # initialize distance function and epsilon
        sample_from_prior = self._prior_sample()

        self.distance_function.initialize(sample_from_prior)

        def distance_to_ground_truth_function(x):
            return self.distance_function(x, self.x_0)

        self.eps.initialize(sample_from_prior,
                            distance_to_ground_truth_function)
        self.history.store_initial_data(ground_truth_model,
                                        abc_options,
                                        observed_summary_statistics,
                                        ground_truth_parameter,
                                        model_names,
                                        self.distance_function.to_json(),
                                        self.eps.to_json(),
                                        self.population_strategy.to_json())

    def _prior_sample(self):
        """
        Only sample from prior and return results without changing
        the history of the Epsilon. This can be used to get initial samples
        for the distance function or the epsilon to calibrate them.

        .. warning::

            The sample is cached.
        """
        if self._points_sampled_from_prior is None:
            def sample_one():
                m = self.model_prior.rvs()
                par = self.parameter_priors[m].rvs()
                return m, par

            def simulate_one(para):
                (m, par) = para
                model_result = self.models[m].summary_statistics(
                    par, self.summary_statistics)
                return model_result.sum_stats

            self._points_sampled_from_prior = (
                self.sampler.sample_until_n_accepted(
                    sample_one, simulate_one, lambda x: True,
                    self.population_strategy.nr_particles))
        return self._points_sampled_from_prior

    def _evaluate_proposal(self, m_ss, theta_ss, current_eps, t,
                           model_probabilities):
        """
        This is where the actual model evaluation happens.
        """
        # from here, theta_ss is valid according to the prior
        distance_list = []
        summary_statistics_list = []
        for _ in range(self.population_strategy.nr_samples_per_parameter):
            model_result = self.models[m_ss].accept(
                theta_ss, self.summary_statistics,
                lambda x: self.distance_function(x, self.x_0), current_eps)
            if model_result.accepted:
                distance_list.append(model_result.distance)
                summary_statistics_list.append(model_result.sum_stats)

        if len(distance_list) > 0:
            weight = self._calc_proposal_weight(
                distance_list, m_ss, theta_ss, t, model_probabilities)
        else:
            weight = 0
        valid_particle = ValidParticle(
            m_ss, theta_ss, weight, distance_list, summary_statistics_list)
        return valid_particle

    def _calc_proposal_weight(self, distance_list, m_ss, theta_ss,
                              t, model_probabilities):
        if t == 0:
            weight = (len(distance_list)
                      / self.population_strategy.nr_samples_per_parameter)
        else:
            model_factor = sum(
                row.p * self.model_perturbation_kernel.pmf(m_ss, m)
                for m, row in model_probabilities.iterrows())
            particle_factor = self.transitions[m_ss].pdf(
                pd.Series(dict(theta_ss)))
            normalization = model_factor * particle_factor
            if normalization == 0:
                print('normalization is zero!')
            # reflects stochasticity of the model
            fraction_accepted_runs_for_single_parameter = (
                len(distance_list)
                / self.population_strategy.nr_samples_per_parameter)
            weight = (self.model_prior.pmf(m_ss)
                      * self.parameter_priors[m_ss].pdf(theta_ss)
                      * fraction_accepted_runs_for_single_parameter
                      / normalization)
        return weight

    def _generate_valid_proposal(self, t, m, p):
        """
        Parameters
        ----------
        t: populaton number
        m: Indices of alive models
        p: Probabilities of alive models

        Returns
        -------

        """
        # first generation
        if t == 0:  # sample from prior
            m_ss = self.model_prior.rvs()
            theta_ss = self.parameter_priors[m_ss].rvs()
            return m_ss, theta_ss

        # later generation
        while True:  # find m_s and theta_ss, valid according to prior
            if len(m) > 1:
                index = fast_random_choice(p)
                m_s = m[index]
                m_ss = self.model_perturbation_kernel.rvs(m_s)
                # theta_s is None if the population m_ss has died out.
                # This can happen since the model_perturbation
                # _kernel can return  a model nr which has died out.
                if m_ss not in m:
                    continue
            else:
                m_ss = m[0]
            theta_ss = self.transitions[m_ss].rvs()

            if (self.model_prior.pmf(m_ss)
                    * self.parameter_priors[m_ss].pdf(theta_ss) > 0):
                return m_ss, theta_ss

    def run(self, minimum_epsilon: float, max_nr_populations: int) -> History:
        """
        Run the ABCSMC model selection until either of the stopping
        criteria is met.

        Parameters
        ----------
        minimum_epsilon: float
            Stop if epsilon is smaller than minimum epsilon specified here.

        max_nr_populations: int
            Tha maximum number of populations. Stop if this number is reached.


        Population after population is sampled and particles which are close
        enough to the observed data are accepted into the next population.
        If an adaptive Epsilon is specified (this is the default), then
        the acceptance threshold decreases from population to population
        automatically in a data dependent way.

        Sampling of further populations is stopped, when either of the two
        stopping criteria is met:

            * the maximum number of populations ``max_nr_populations``
              is reached
            * or the acceptance threshold for the last sampled population was
              smaller than ``minimum_epsilon``.

        The value of ``minimum_epsilon`` determines the quality of the ABCSMC
        approximation. The smaller the better. But sampling time also increases
        with decreasing ``minimum_epsilon``.

        This method can be called repeatedly to sample further populations
        after sampling was stopped once.
        """
        t0 = self.history.max_t + 1
        self.history.start_time = datetime.datetime.now()
        # not saved as attribute b/c Mapper of type
        # "ipython_cluster" is not pickable
        for t in range(t0, t0+max_nr_populations):
            # this is calculated here to avoid double initialization of medians
            current_eps = self.eps(t, self.history)
            abclogger.info('t:' + str(t) + ' eps:' + str(current_eps))
            self._fit_transitions(t)
            self._adapt_population(t)
            # cache model_probabilities to not to query the database so soften
            model_probabilities = self.history.get_model_probabilities(
                self.history.max_t)
            abclogger.debug('now submitting population ' + str(t))

            m = sp.array(model_probabilities.index)
            p = sp.array(model_probabilities.p)

            def sample_one():
                return self._generate_valid_proposal(t, m, p)

            def eval_one(par):
                return self._evaluate_proposal(*par, current_eps, t,
                                               model_probabilities)

            def accept_one(particle):
                return len(particle.distance_list) > 0

            population = self.sampler.sample_until_n_accepted(
                sample_one, eval_one, accept_one,
                self.population_strategy.nr_particles)

            population = [particle for particle in population
                          if not isinstance(particle, Exception)]
            abclogger.debug('population ' + str(t) + ' done')
            self.history.append_population(
                t, current_eps, population, self.sampler.nr_evaluations_)
            abclogger.debug(
                '\ntotal nr simulations up to t =' + str(t) + ' is '
                + str(self.history.total_nr_simulations))

            if (current_eps <= minimum_epsilon or
                (self.stop_if_only_single_model_alive
                 and self.history.nr_of_models_alive() <= 1)):
                break
        self.history.done()
        return self.history

    def _adapt_population(self, t):
        if t == 0:  # we need a particle population to do the fitting
            return

        w = self.history.get_model_probabilities(
            self.history.max_t)["p"].as_matrix()
        # make a copy in case the population strategy messes with
        # the transitions
        # WARNING: the deepcopy also copies the random states of scipy.stats
        # distributions
        copied_transitions = copy.deepcopy(self.transitions)
        self.population_strategy.adapt_population_size(copied_transitions, w)

    def _fit_transitions(self, t):
        if t == 0:  # we need a particle population to do the fitting
            return

        for m in self.history.alive_models(t - 1):
            particles, w = self.history.get_distribution(m, t - 1)
            self.transitions[m].fit(particles, w)
