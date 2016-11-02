"""
Main ABC algorithm
------------------
"""

import datetime
import sys
import time
from typing import List, Callable, Iterable, Any, TypeVar
from pyabc.workspace_dennis.map_wrapper import MapWrapperDefault

model_output = TypeVar("model_output")

from .model import Model
from .parameters import Parameter
from .random_variables import RV, ModelPerturbationKernel, Distribution, Kernel
from .distance_functions import DistanceFunction
from .epsilon import Epsilon
from .storage import History


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

    model_prior_distribution: RV
        A random variable giving the prior weights of the model classes.
        If the prior is uniform over the model classes
        this is something like ``RV("randint", 0, len(models))``.

    model_perturbation_kernel: ModelPerturbationKernel
        Kernel which governs with which probability to switch the model
        for a given sample.

    parameter_given_model_prior_distribution: List[Distribution]
        A list of prior distributions for the models' parameters.
        Each list entry is the prior distribution for the corresponding model.

    adaptive_parameter_perturbation_kernels: List[Callable[[int, dict], Kernel]]
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

    debug: bool
        Whether to output additional debug information

    max_nr_allowed_sample_attempts_per_particle: int
        The maximum number of sample attempts allowed for each particle.
        If this number is reached, the sampling for a particle is stopped.
        Hence, a population may return with less particles than started.
        This is an approximation to the ABCSMC algorithm which ensures, that
        the algorithm terminates.

    min_nr_particles_per_population: int
        Minimum number of samples which have to be accepted for a population.
        If this number is not reached, the algorithm stops.
        This option, together with the ``max_nr_allowed_sample_attempts_per_particle``
        ensures that the algorithm terminates.

        More precisely, this parameter determines to which extend an approximation to the
        ABCSMC algorithm is allowed.


    .. [#toni-stumpf] Toni, Tina, and Michael P. H. Stumpf.
                  “Simulation-Based Model Selection for Dynamical
                  Systems in Systems and Population Biology.”
                  Bioinformatics 26, no. 1 (January 1, 2010):
                  104–10. doi:10.1093/bioinformatics/btp619.
    """
    def __init__(self,
                 models: List[Model],
                 model_prior_distribution: RV,
                 model_perturbation_kernel: ModelPerturbationKernel,
                 parameter_given_model_prior_distribution: List[Distribution],
                 adaptive_parameter_perturbation_kernels: List[Callable[[int, dict], Kernel]],
                 distance_function: DistanceFunction,
                 eps: Epsilon,
                 nr_particles: int,
                 mapper=map,
                 map_wrapper=None,
                 debug: bool =False,
                 max_nr_allowed_sample_attempts_per_particle: int =500,
                 min_nr_particles_per_population: int =1,
                 summary_statistics: Callable[[model_output], dict]=identity):

        # sanity checks
        self.models = list(models)
        if not (len(self.models)
                == len(parameter_given_model_prior_distribution)
                == len(adaptive_parameter_perturbation_kernels)):
            raise Exception("Nr of models has to be equal to the number of parameter prior distributions has to be equal"
                            " to the number of parameter perturbation kernels")

        ## DR START
        #self.mapper = mapper
        if map_wrapper is None:
            self.mapper = MapWrapperDefault(mapper)
        else:
            self.mapper = map_wrapper
            self.mapper.set_map_fun(mapper)
        ## DR END


        self.model_prior_distribution = model_prior_distribution
        self.model_perturbation_kernel = model_perturbation_kernel
        self.parameter_given_model_prior_distribution = parameter_given_model_prior_distribution  # this cannot be serialized by dill
        self.adaptive_parameter_perturbation_kernels = adaptive_parameter_perturbation_kernels
        self.distance_function = distance_function
        self.eps = eps
        self.summary_statistics = summary_statistics
        self.nr_particles = nr_particles
        self.debug = debug
        self.stop_if_only_single_model_alive = True
        self.x_0 = None
        self.history = None  # History
        self._points_sampled_from_prior = None
        self.max_nr_allowed_sample_attempts_per_particle = max_nr_allowed_sample_attempts_per_particle
        self.min_nr_particles_per_population = min_nr_particles_per_population

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
                 abc_options: dict,
                 model_names: Iterable[str]):
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
        self.history = History(abc_options['db_path'], len(self.models), list(model_names),
                               self.min_nr_particles_per_population)

        # initialize distance function and epsilon
        sample_from_prior = self.sample_from_prior()
        self.distance_function.initialize(sample_from_prior)

        def distance_to_ground_truth_function(x):
            return self.distance_function(x, self.x_0)

        self.eps.initialize(sample_from_prior, distance_to_ground_truth_function)
        self.history.store_initial_data(ground_truth_model, abc_options,
                                        observed_summary_statistics, ground_truth_parameter,
                                        self.distance_function.to_json(),
                                        self.eps.to_json())

    def sample_from_prior(self) -> List[dict]:
        """
        Only sample from prior and return results without changing
        the history of the Epsilon. This can be used to get initial samples
        for the distance function or the epsilon to calibrate them.

        .. warning::

            The sample is cached.
        """
        if self._points_sampled_from_prior is None:
            # not saved as attribute b/c Mapper of type "ipython_cluster" is not pickable
            def sample(_):
                m = self.model_prior_distribution.rvs()
                par = self.parameter_given_model_prior_distribution[m].rvs()
                model_result = self.models[m].summary_statistics(par, self.summary_statistics)
                return model_result.sum_stats
            if self.debug:
                print('sample from prior')
            self._points_sampled_from_prior = list(self.mapper.wrap_map(sample, list(range(self.nr_particles))))
            if self.debug:
                print('sample from prior done')
        if self.debug:
            print('return sample from prior')
        return self._points_sampled_from_prior

    def _sample_single_particle(self, parameter_perturbation_kernels,
                               nr_samples_per_particle: int,
                               t: int,
                               t0: int,
                               current_eps: float) -> (int, Parameter, float, List[float], int, List[dict]):
        """
        This is where the actual model evaluation happens.
        The core ABCSMC algorithm is also implemented here.

        Parameters
        ----------
        parameter_perturbation_kernels
        nr_samples_per_particle
        t: int
            population number
        t0: int
            initial population
        current_eps

        Returns
        -------

        """
        simulation_counter = 0
        while True:  # find valid theta_ss and (corresponding b) according to data x_0
            m_ss, theta_ss = self.generate_valid_proposal(parameter_perturbation_kernels, t)
            # from here, theta_ss is valid according to the prior
            distance_list = []
            summary_statistics_list = []
            for __ in range(nr_samples_per_particle[t-t0]):
                ##### MODEL SIMULATION - THIS IS THE EXPENSIVE PART ######
                simulation_counter += 1
                # stop builder if it takes too long
                if simulation_counter > self.max_nr_allowed_sample_attempts_per_particle:
                    print("Max nr of samples (={n_max}) for particle reached."
                          .format(n_max=self.max_nr_allowed_sample_attempts_per_particle), file=sys.stderr)
                    return None
                start_time = time.time()
                model_result = self.models[m_ss].accept(theta_ss, self.summary_statistics,
                                                        lambda x: self.distance_function(x, self.x_0), current_eps)
                if model_result.accepted:
                    distance_list.append(model_result.distance)
                    summary_statistics_list.append(model_result.sum_stats)

                end_time = time.time()
                duration = end_time - start_time
                if self.debug:
                    print("Sampled model={}-{}, delta_time={}s, end_time={},  theta_ss={}"
                          .format(m_ss, self.history.model_names[m_ss], duration, end_time,
                                  theta_ss))
            if len(distance_list) > 0:
                break

        if t == 0:
            weight = len(distance_list) / nr_samples_per_particle[t-t0]
        else:
            normalization = (sum(self.history.get_model_probabilities(t-1)[j] * self.model_perturbation_kernel.pmf(m_ss, j)
                                 for j in range(len(self.models)))
                             * sum(particle['weight']  # this is already conditioned on m,
                                                       # so do not divide by P_{t-1}(m_{t-1} = m_t^{(i)}) = P_{t-1}(m_t^{(i)})
                                   * parameter_perturbation_kernels[m_ss].pdf(theta_ss, particle['parameter'])
                                   for particle in self.history.store[t-1][m_ss])
                            )
            if normalization == 0:
                print('normalization is zero!')
            weight = (self.model_prior_distribution.pmf(m_ss)
                      * self.parameter_given_model_prior_distribution[m_ss].pdf(theta_ss)
                      * len(distance_list) / nr_samples_per_particle[t-t0]
                      / normalization)
        if self.debug:
            print('.', end='')
        return m_ss, theta_ss, weight, distance_list, simulation_counter, summary_statistics_list

    def generate_valid_proposal(self, parameter_perturbation_kernels, t):
        while True:  # find m_s and theta_ss, valid according to prior
            if t == 0:  # sample from prior
                m_ss = self.model_prior_distribution.rvs()
                theta_ss = self.parameter_given_model_prior_distribution[m_ss].rvs()
            else:
                m_s = self.history.sample_from_models(t - 1)
                m_ss = self.model_perturbation_kernel.rvs(m_s)
                theta_s = self.history.sample_from_population(t - 1, m_ss)
                if theta_s is not None:  # theta_s is None if the population m_s has died out
                    theta_ss = parameter_perturbation_kernels[m_ss].rvs(theta_s)
                else:
                    theta_ss = None
            if theta_ss is not None and (self.model_prior_distribution.pmf(m_ss)
                                             * self.parameter_given_model_prior_distribution[m_ss].pdf(theta_ss) > 0):
                return m_ss, theta_ss

    def run(self, nr_samples_per_particle: List[int], minimum_epsilon: float) -> History:
        """
        Run the ABCSMC model selection. This method can be called many times. It makes another
        step continuing where it has stopped before.

        It is stopped when the maximum number of populations is reached
        or the ``minimum_epsilon`` value is reached.

        Parameters
        ----------

        nr_samples_per_particle: List[int]
            The length of the list determines the maximal number of populations.

            The entries of the list the number of iterated simulations
            in the notation from Toni et al 2009 these are the :math:`B_t`.
            Usually, the entries are all ones, e.g. in most cases you'll have:
            ``nr_samples_per_particle = [1] * nr_populations``.

        minimum_epsilon: float
            Stop if epsilon is smaller than minimum epsilon specified here.
        """
        t0 = self.history.t
        self.history.start_time = datetime.datetime.now()
        # not saved as attribute b/c Mapper of type "ipython_cluster" is not pickable
        for t in range(t0, t0+len(nr_samples_per_particle)):
            current_eps = self.eps(t, self.history)  # this is calculated here to avoid double initialization of medians
            if self.debug:
                print('t:', t, 'eps:', current_eps)
            statistics = self.history.get_statistics(t-1)
            parameter_perturbation_kernels = self._make_parameter_perturbation_kernels(statistics, t)
            if self.debug:
                print('now submitting population', t)
            new_particle_population = list(
                    self.mapper.wrap_map(lambda _: self._sample_single_particle(parameter_perturbation_kernels,
                                                                    nr_samples_per_particle,
                                                                    t,
                                                                    t0,
                                                                    current_eps),
                                [None] * self.nr_particles))
            new_particle_population = [particle for particle in new_particle_population
                                       if not isinstance(particle, Exception)]
            if self.debug:
                print('population', t, 'done')
            new_particle_population_non_empty = self.history.append_population(t, current_eps, new_particle_population)
            if self.debug:
                print('\ntotal nr simulations up to t =', t, 'is', self.history.total_nr_simulations)
                sys.stdout.flush()
            if (not new_particle_population_non_empty or
                (current_eps <= minimum_epsilon) or
                (self.stop_if_only_single_model_alive and self.history.nr_of_models_alive() <= 1)):
                break
        self.history.done()
        return self.history

    def _make_parameter_perturbation_kernels(self, statistics, t):
        parameter_perturbation_kernels = [apk(t, stat) if stat is not None else None
                                          for apk, stat in
                                          zip(self.adaptive_parameter_perturbation_kernels, statistics)]
        return parameter_perturbation_kernels
