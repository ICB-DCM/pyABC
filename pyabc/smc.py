"""
Parallel Approximate Bayesian Computation - Sequential Monte Carlo
==================================================================

The ABCSMC class is the most central class of the pyABC package.
Most of the other classes serve to configure it. (I.e. the other classes
implement a Strategy pattern.)
"""

import datetime
import logging
from typing import List, Callable, TypeVar
import pandas as pd
import scipy as sp
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
from .platform_factory import DefaultSampler
import copy
import warnings

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

    models: list of models, single model, single function or list of functions
       * If models is a function, then the function should have a single
         parameter, which is of dictionary type, and should return a single
         dictionary, which contains the simulated data.
       * If models is a list of functions, then the first point applies to
         each function.
       * Models can also be a list of Model instances or a single
         Model instance.

       This model's output is passed to the summary statistics calculation.
       Per default, the model is assumed to already return the calculated
       summary statistcs. Accordingly, the default summary_statistics
       function is just the identity.

    parameter_priors: List[Distribution]
        A list of prior distributions for the models' parameters.
        Each list entry is the prior distribution for the corresponding model.

    distance_function: DistanceFunction, optional
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

    model_prior: RV, optional
        A random variable giving the prior weights of the model classes.
        The default is a uniform prior over the model classes,
        ``RV("randint", 0, len(models))``.

    model_perturbation_kernel: ModelPerturbationKernel
        Kernel which governs with which probability to switch from one
        model to anoter model for a given sample while generating proposals
        for the subsequent population from the current population.

    transitions: List[Transition], Transition, optional
        A list of :class:`pyabc.transition.Transition` objects
        or a single :class:`pyabc.transition.Transition` in case
        of a single model. Defaults to multivariate normal transitions for
        every model.

    eps: Epsilon, optional
        Accepts any :class:`pyabc.epsilon.Epsilon` subclass.
        The default is the :class:`pyabc.epsilon.MedianEpsilon` which adapts
        automatically. The object passed here determines how the acceptance
        threshold scheduling is performed.

    sampler: Sampler, optional
        In some cases, a mapper implementation will require initialization
        to run properly, e.g. database connection, grid setup, etc..
        The sampler is an object that encapsulates this information.
        The default sampler :class:`pyabc.sampler.MulticoreEvalParallelSampler`
        will parallelize across the cores of a single
        machine only.


    Attributes
    ----------

    stop_if_only_single_model_alive: bool
        Defaults to False. Set this to true if you want to stop ABCSMC
        automatically as soon as only a single model has survived.


    .. [#tonistumpf] Toni, Tina, and Michael P. H. Stumpf.
                  “Simulation-Based Model Selection for Dynamical
                  Systems in Systems and Population Biology.”
                  Bioinformatics 26, no. 1 (January 1, 2010):
                  104–10. doi:10.1093/bioinformatics/btp619.
    """
    def __init__(self, models: Union[List[Model], Model],
                 parameter_priors: Union[List[Distribution],
                                         Distribution, Callable],
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
            self.sampler = DefaultSampler()
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
        warnings.warn("This method is deprecated and removed "
                      "in pyABC 10.0.", DeprecationWarning, stacklevel=2)
        self.stop_if_only_single_model_alive = False

    def set_data(self, observed_summary_statistics: dict,
                 abc_options: Union[dict, str],
                 ground_truth_model: int = -1,
                 ground_truth_parameter: dict = None):
        """
        This method is an alias for the ``new`` method.
        This method is deprecated and to be removed in future releases.
        Use the ``new`` method instead.
        Note that the argument order has changed!

        .. warning::

           The method "set_data" is deprecated.
        """
        warnings.warn("The method \"set_data\" is deprecated "
                      "and to be removed "
                      "in pyABC 0.10.0. "
                      "Use the method \"new\" instead. "
                      "Note that the API has changed slightly!",
                      DeprecationWarning, stacklevel=2)
        if not isinstance(abc_options, str):
            db = abc_options["db_path"]
            meta = abc_options
        else:
            db = abc_options
            meta = {}
        return self.new(db,
                        observed_sum_stat=observed_summary_statistics,
                        gt_model=ground_truth_model,
                        gt_par=ground_truth_parameter,
                        meta_info=meta)

    def load(self, db: str, abc_id: int = 1):
        """
        Load an ABC-SMC run for continuation.

        Parameters
        ----------
        db: str
            A SQLAlchemy database identifier pointing to the database from
            which to continue a run.

        abc_id: int, optional
            The id of the ABC-SMC run in the database which is to be continued.
            The default is 1. If more than one ABC-SMC run is stored, use
            the ``abc_id`` parameter to indicate which one to continue.


        .. note::

            The Epsilon's and distance function's initialize methods are
            not called when an ABCSMC run is loaded.
        """
        self.history = History(db)
        self.history.id = abc_id
        self.x_0 = self.history.observed_sum_stat()

    def new(self, db: str,
            observed_sum_stat: dict = None,
            *,
            gt_model: int = None,
            gt_par: dict = None,
            meta_info=None):
        """
        Make a new ABCSMC run.

        Parameters
        ----------

        db: str
            Has to be a valid SQLAlchemy database identifier.
            This indicates the database to be used (and created if necessary
            and possible) for the ABC-SMC run.

        observed_sum_stat : dict, optional
               This is the really important parameter here. It is of the
               form ``{'statistic_1': val_1, 'statistic_2': val_2, ... }``.

               The dictionary provided here represents the measured data.
               Particle during ABCSMC sampling are compared against the
               summary statistics provided here.

               This parameter is optional, as the distance function might
               implement comparison to the observed data on its own.
               Not giving this parameter is equivalent to passing an empty
               dictionary ``{}``.

        gt_model: int, optional
            This is only meta data stored to the database, but not actually
            used for the ABCSMC algorithm If you want to predict your ABCSMC
            procedure against synthetic samples, you can use
            this parameter to indicate the ground truth model number.
            This helps with futher analysis. If you use actually measured data
            (and don't know the ground truth) you don't have to set this.

        gt_par: dict, optional
            Similar to ``ground_truth_model``, this is only for recording
            purposes in the database, but not used in the ABCSMC algorithm.
            This stores the parameters of the ground truth model
            if it was synthetically obtained.
            Don't give this parameter if you don't know the ground truth.

        meta_info: dict, optional
            Can contain an arbitrary number of keys, only for recording
            purposes. Store arbitrary
            meta information in this dictionary. Can be used for really
            anything.
            This dictionary is stored in the database.
        """

        # initialize
        if observed_sum_stat is None:
            observed_sum_stat = {}

        self.x_0 = observed_sum_stat
        model_names = [model.name for model in self.models]

        self.history = History(db)

        if gt_par is None:
            gt_par = {}

        self._initialize_dist_and_eps()
        self.history.store_initial_data(gt_model,
                                        meta_info,
                                        observed_sum_stat,
                                        gt_par,
                                        model_names,
                                        self.distance_function.to_json(),
                                        self.eps.to_json(),
                                        self.population_strategy.to_json())
        return self.history.id

    def _initialize_dist_and_eps(self):
        # initialize distance function and epsilon
        self.distance_function.initialize(self._prior_sample())

        def distance_to_ground_truth_function(x):
            return self.distance_function(x, self.x_0)

        self.eps.initialize(self._prior_sample(),
                            distance_to_ground_truth_function)

    def _prior_sample(self) -> List[dict]:
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
                m, par = para
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

    def run(self, minimum_epsilon: float, max_nr_populations: int,
            min_acceptance_rate: float = 0., **kwargs) -> History:
        """
        Run the ABCSMC model selection until either of the stopping
        criteria is met.

        Parameters
        ----------

        minimum_epsilon: float
            Stop if epsilon is smaller than minimum epsilon specified here.

        max_nr_populations: int
            The maximum number of populations. Stop if this number is reached.

        min_acceptance_rate: float, optional
            Minimal allowed acceptance rate. Sampling stops if a population
            has a lower rate.


        Population after population is sampled and particles which are close
        enough to the observed data are accepted and added to the next
        population.
        If an adaptive Epsilon is specified (this is the default), then
        the acceptance threshold decreases from population to population
        automatically in a data dependent way.

        Sampling of further populations is stopped, when either of the three
        stopping criteria is met:

            * the maximum number of populations ``max_nr_populations``
              is reached,
            * the acceptance threshold for the last sampled population was
              smaller than ``minimum_epsilon``,
            * or the acceptance rate dropped below ``acceptance_rate``.

        The value of ``minimum_epsilon`` determines the quality of the ABCSMC
        approximation. The smaller the better. But sampling time also increases
        with decreasing ``minimum_epsilon``.

        This method can be called repeatedly to sample further populations
        after sampling was stopped once.
        """
        if len(kwargs) > 1:
            raise TypeError("Keyword arguments are not allowed.")

        if "acceptance_rate" in kwargs:
            warnings.warn("The acceptance_rate argument is deprecated and "
                          "removed in pyABc 0.9.0. "
                          "Use min_acceptance_rate instead.",
                          DeprecationWarning, stacklevel=2)
            min_acceptance_rate = kwargs["acceptance_rate"]

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
            nr_evaluations = self.sampler.nr_evaluations_
            model_names = [model.name for model in self.models]
            self.history.append_population(
                t, current_eps, population, nr_evaluations,
                model_names)
            abclogger.debug(
                '\ntotal nr simulations up to t =' + str(t) + ' is '
                + str(self.history.total_nr_simulations))

            current_acceptance_rate = len(population) / nr_evaluations
            if (current_eps <= minimum_epsilon
               or (self.stop_if_only_single_model_alive
                   and self.history.nr_of_models_alive() <= 1)
               or current_acceptance_rate < min_acceptance_rate):
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
