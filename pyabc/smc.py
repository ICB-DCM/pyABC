"""
Parallel Approximate Bayesian Computation - Sequential Monte Carlo
==================================================================


"""

import datetime
import logging
from typing import List, Callable, TypeVar

abclogger = logging.getLogger("ABC")
import pandas as pd
import scipy as sp

from .parallel import MulticoreSampler
from .distance_functions import DistanceFunction, to_distance
from .epsilon import Epsilon, MedianEpsilon
from .model import Model
from .parameters import ValidParticle
from .transition import Transition, MultivariateNormalTransition
from .random_variables import RV, ModelPerturbationKernel, Distribution
from .storage import History
from .populationstrategy import PopulationStrategy, ConstantPopulationStrategy
from .random import fast_random_choice
from typing import Union


model_output = TypeVar("model_output")


def identity(x):
    return x





class ABCSMC:
    """
    Approximate Bayesian Computation - Sequential Monte Carlo (ABCSMC).

    This is an implementation of an ABCSMC algorithm similar to [#toni-stumpf]_


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

    parameter_priors: List[Distribution]
        A list of prior distributions for the models' parameters.
        Each list entry is the prior distribution for the corresponding model.

    transitions: List[Callable[[int, dict], Kernel]]
        A list of functions mapping ``(t, stat) -> Kernel``, where

            * ``t`` is the population nr
            * ``stat`` a dictionary of summary statistics.
               E.g. ``stat['std']['parameter_1']`` is the standard deviation of ``parameter_1``.

                .. warning:: If a model has only one particle left the standdardeviation is zero.

        This callable is called at the beginning of a new population with the statistics dictionary
        from the last population to determine the new parameter perturbation kernel for the next population.

    distance_function: DistanceFunction
        Measures the distance of the tentatively sampled particle to the measured data.

    eps: Epsilon
        Returns the current acceptance epsilon.
        This epsilon changes from population to population.
        The eps instance provided the strategy fo how to change it.

    mapper: map like
        Something like the built in map.
        I.e. mapper(f, args) takes a callable ``f`` and applies it the the arguments in the list ``args``.
        This mapper is used for particle sampling.
        It can be a distributed mapper such as the :class:`parallel.sge.SGE` class.

    sampler:
        In some cases, a mapper implementation will require initialization to run properly,
        e.g. database connection, grid setup, etc... The sampler is an object that encapsulates
        this information.  The default sampler will simply call the callable mapper at the right
        place; a more involved sampler will help the mapper-function to distribute function calls
        accross a distributed infrastructure.

    debug: bool
        Whether to output additional debug information



    .. [#toni-stumpf] Toni, Tina, and Michael P. H. Stumpf.
                  “Simulation-Based Model Selection for Dynamical
                  Systems in Systems and Population Biology.”
                  Bioinformatics 26, no. 1 (January 1, 2010):
                  104–10. doi:10.1093/bioinformatics/btp619.
    """
    def __init__(self, models: List[Model],
                 parameter_priors: List[Distribution],
                 distance_function,
                 population_strategy: Union[PopulationStrategy, int],
                 summary_statistics: Callable[[model_output], dict] = identity,
                 model_prior: RV = None,
                 model_perturbation_kernel: ModelPerturbationKernel = None,
                 transitions: List[Transition] = None, eps: Epsilon = None,
                 sampler=None):

        # sanity checks
        self.models = list(models)

        self.parameter_priors = parameter_priors  # this cannot be serialized by dill
        assert len(self.models) == len(self.parameter_priors), "Number models and number parameter priors have to agree"

        self.distance_function = to_distance(distance_function)

        self.summary_statistics = summary_statistics

        if model_prior is None:
            model_prior = RV("randint", 0, len(self.models))
        self.model_prior = model_prior

        if model_perturbation_kernel is None:
            model_perturbation_kernel = ModelPerturbationKernel(len(self.models), probability_to_stay=.7)
        self.model_perturbation_kernel = model_perturbation_kernel

        if transitions is None:
            transitions = [MultivariateNormalTransition() for _ in self.models]
        self.transitions = transitions  # type: List[Transition]

        if eps is None:
            eps = MedianEpsilon()
        self.eps = eps

        self.population_strategy = population_strategy

        if sampler is None:
            self.sampler = MulticoreSampler()
        else:
            self.sampler = sampler

        self.stop_if_only_single_model_alive = True
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
                 ground_truth_model: int,
                 ground_truth_parameter: dict,
                 abc_options: dict):
        """
        Set the data to be fitted.

        Parameters
        ----------

        observed_summary_statistics : dict
               **This is the really important parameter here**. It is of the form
               ``{'statistic_1' : val_1, 'statistic_2': val_2, ... }``.

               The dictionary provided here represents the measured data.
               Particle during ABCSMC sampling are compared against the summary statistics
               provided here.

        ground_truth_model: int
            This is only meta data stored to the database, but not actually used for the ABCSMC algorithm
            If you want to predict your ABCSMC procedure against synthetic samples, you can use
            this parameter to indicate the ground truth model number. This helps with futher analysis.
            If you use actually measured data (and don't know the ground truth) you can set this to anything.
            A value if ``-1`` is recommended.

        ground_truth_parameter: dict
            Similar to ``ground_truth_model``, this is only for recording purposes, but not used in the
            ABCSMC algorithm. This stores the parameters of the ground truth model if it was synthetically
            obtained.

        abc_options: dict
            Has to contain the key "db_path" which has to be a valid SQLAlchem database identifier.
            Can caontain an arbitrary number of additional keys, only for recording purposes.
            Store arbitrary meta information in this dictionary. Can be used for really anything.

        model_names: List[str]
            Only for recording purposes. Record names of the models
        """
        # initialize
        self.x_0 = observed_summary_statistics
        model_names = [model.name for model in self.models]
        self.history = History(abc_options['db_path'])

        # initialize distance function and epsilon
        sample_from_prior = self.prior_sample()

        self.distance_function.initialize(sample_from_prior)

        def distance_to_ground_truth_function(x):
            return self.distance_function(x, self.x_0)

        self.eps.initialize(sample_from_prior, distance_to_ground_truth_function)
        self.history.store_initial_data(ground_truth_model,
                                        abc_options,
                                        observed_summary_statistics,
                                        ground_truth_parameter,
                                        model_names,
                                        self.distance_function.to_json(),
                                        self.eps.to_json(),
                                        self.population_strategy.to_json())

    def prior_sample(self):
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
                model_result = self.models[m].summary_statistics(par, self.summary_statistics)
                return model_result.sum_stats

            sample_from_prior = self.sampler.sample_until_n_accepted(sample_one, simulate_one, lambda x: True,
                                                                     self.population_strategy.nr_particles)
        else:
            sample_from_prior = self._points_sampled_from_prior
        return sample_from_prior

    def evaluate_proposal(self, m_ss, theta_ss, current_eps, t, model_probabilities):
        """
        This is where the actual model evaluation happens.
        """
        # from here, theta_ss is valid according to the prior
        distance_list = []
        summary_statistics_list = []
        for _ in range(self.population_strategy.nr_samples_per_parameter):
            model_result = self.models[m_ss].accept(theta_ss, self.summary_statistics,
                                                    lambda x: self.distance_function(x, self.x_0), current_eps)
            if model_result.accepted:
                distance_list.append(model_result.distance)
                summary_statistics_list.append(model_result.sum_stats)

        if len(distance_list) > 0:
            weight = self.calc_proposal_weight(distance_list, m_ss, theta_ss, t, model_probabilities)
        else:
            weight = 0
        valid_particle = ValidParticle(m_ss, theta_ss, weight, distance_list, summary_statistics_list)
        return valid_particle

    def calc_proposal_weight(self, distance_list, m_ss, theta_ss, t, model_probabilities):
        if t == 0:
            weight = len(distance_list) / self.population_strategy.nr_samples_per_parameter
        else:
            model_factor = sum(row.p * self.model_perturbation_kernel.pmf(m_ss, m)
                                 for m, row in model_probabilities.iterrows())
            particle_factor = self.transitions[m_ss].pdf(pd.Series(dict(theta_ss)))
            normalization = model_factor * particle_factor
            if normalization == 0:
                print('normalization is zero!')
            fraction_accepted_runs_for_single_parameter = len(distance_list) / self.population_strategy.nr_samples_per_parameter  # reflects stochasticity of the model
            weight = (self.model_prior.pmf(m_ss)
                      * self.parameter_priors[m_ss].pdf(theta_ss)
                      * fraction_accepted_runs_for_single_parameter
                      / normalization)
        return weight

    def generate_valid_proposal(self, t, m, p):
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
                # This can happen since the model_perturbation_kernel can return
                # a model nr which has died out.
                if m_ss not in m:
                    continue
            else:
                m_ss = m[0]
            theta_ss = self.transitions[m_ss].rvs()

            if (self.model_prior.pmf(m_ss)
                                             * self.parameter_priors[m_ss].pdf(theta_ss) > 0):
                return m_ss, theta_ss

    def run(self, minimum_epsilon: float) -> History:
        """
        Run the ABCSMC model selection. This method can be called many times. It makes another
        step continuing where it has stopped before.

        It is stopped when the maximum number of populations is reached
        or the ``minimum_epsilon`` value is reached.

        Parameters
        ----------

        minimum_epsilon: float
            Stop if epsilon is smaller than minimum epsilon specified here.
        """
        t0 = self.history.max_t + 1
        self.history.start_time = datetime.datetime.now()
        # not saved as attribute b/c Mapper of type "ipython_cluster" is not pickable
        for t in range(t0, t0+self.population_strategy.nr_populations):
            current_eps = self.eps(t, self.history)  # this is calculated here to avoid double initialization of medians
            abclogger.debug('t:' + str(t) + ' eps:' + str(current_eps))
            self.fit_transitions(t)
            # cache model_probabilities to not to query the database so soften
            model_probabilities = self.history.get_model_probabilities(self.history.max_t)
            abclogger.debug('now submitting population ' + str(t))

            m = sp.array(model_probabilities.index)
            p = sp.array(model_probabilities.p)

            def sample_one():
                return self.generate_valid_proposal(t, m, p)

            def eval_one(par):
                return self.evaluate_proposal(*par, current_eps, t, model_probabilities)

            def accept_one(particle):
                return len(particle.distance_list) > 0

            population = self.sampler.sample_until_n_accepted(sample_one, eval_one, accept_one,
                                                              self.population_strategy.nr_particles)

            population = [particle for particle in population if not isinstance(particle, Exception)]
            abclogger.debug('population ' + str(t) + ' done')
            nr_particles_in_this_population = sum(1 for p in population if p is not None)
            enough_particles = nr_particles_in_this_population >= self.population_strategy.min_nr_particles()
            if enough_particles:
                self.history.append_population(t, current_eps, population, self.sampler.nr_evaluations_)
            else:
                abclogger.info("Not enough particles in population: Found {f}, required {r}."
                               .format(f=nr_particles_in_this_population, r=self.population_strategy.min_nr_particles()))
            abclogger.debug('\ntotal nr simulations up to t =' + str(t) + ' is ' + str(self.history.total_nr_simulations))

            if (not enough_particles or (current_eps <= minimum_epsilon) or
                (self.stop_if_only_single_model_alive and self.history.nr_of_models_alive() <= 1)):
                break
        self.history.done()
        return self.history

    def fit_transitions(self, t):
        if t == 0:  # we need a particle population to do the fitting
            return

        for m in self.history.alive_models(t - 1):
            particles_df, weights = self.history.weighted_parameters_dataframe(t - 1, m)
            self.transitions[m].fit(particles_df, weights)

        self.population_strategy.adapt_population_size(self.transitions,
                                        self.history.get_model_probabilities(self.history.max_t))
