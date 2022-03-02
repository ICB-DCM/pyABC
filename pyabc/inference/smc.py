"""ABC-SMC"""

import copy
import logging
from datetime import datetime, timedelta
from typing import Callable, List, TypeVar, Union

import numpy as np

from pyabc.acceptor import (
    Acceptor,
    FunctionAcceptor,
    StochasticAcceptor,
    UniformAcceptor,
)
from pyabc.distance import (
    Distance,
    FunctionDistance,
    PNormDistance,
    StochasticKernel,
)
from pyabc.epsilon import Epsilon, MedianEpsilon, TemperatureBase
from pyabc.inference_util import (
    AnalysisVars,
    create_analysis_id,
    create_prior_pdf,
    create_simulate_from_prior_function,
    create_simulate_function,
    create_transition_pdf,
    eps_from_hist,
    termination_criteria_fulfilled,
)
from pyabc.model import FunctionModel, Model
from pyabc.platform_factory import DefaultSampler
from pyabc.population import Population, Sample
from pyabc.populationstrategy import ConstantPopulationSize, PopulationStrategy
from pyabc.random_variables import RV, Distribution
from pyabc.sampler import Sampler
from pyabc.storage import History
from pyabc.transition import (
    ModelPerturbationKernel,
    MultivariateNormalTransition,
    Transition,
)
from pyabc.weighted_statistics import effective_sample_size

logger = logging.getLogger("ABC")

model_output = TypeVar("model_output")


def identity(x):
    return x


def run_cleanup(run):
    """Wrapper: Run and in any case clean up afterwards."""

    def wrapped_run(self: "ABCSMC", *args, **kwargs):
        try:
            # the actual run
            ret = run(self, *args, **kwargs)
        finally:
            # close session and store end time
            self.history.done(end_time=datetime.now())
            # tell samplers to stop any ongoing processes
            self.sampler.stop()
        return ret

    return wrapped_run


class ABCSMC:
    """
    Approximate Bayesian Computation - Sequential Monte Carlo (ABCSMC).

    This is an implementation of an ABCSMC algorithm similar to
    [#tonistumpf]_.

    The ABCSMC class is the most central class of the pyABC package.
    Most of the other classes serve to configure it (i.e. the other classes
    implement a strategy pattern).

    Parameters
    ----------
    models:
        Can be a list of models, a single model, a list of functions, or a
        single function.

        * If models is a function, then the function should have a single
          parameter, which is of dictionary type, and should return a single
          dictionary, which contains the simulated data.
        * If models is a list of functions, then the first point applies to
          each function.
        * Models can also be a list of Model instances or a single
          Model instance.

        This model's output is passed to the summary statistics calculation.
        Per default, the model is assumed to already return the calculated
        summary statistics. Accordingly, the default summary_statistics
        function is just the identity. Note that the sampling and evaluation of
        particles happens in the model's methods, so overriding these offers a
        great deal of flexibility, in particular the freedom to use or ignore
        the distance_function, summary_statistics, and eps parameters here.
    parameter_priors:
        A list of prior distributions for the models' parameters.
        Each list entry is the prior distribution for the corresponding model.
    distance_function:
        Measures the distance of the tentatively sampled particle to the
        measured data.
    population_size:
        Specify the size of the population.
        If ``population_specification`` is an ``int``, then the size is
        constant. Adaptive population sizes are also possible by passing a
        :class:`pyabc.populationstrategy.PopulationStrategy` object.
        The default is 100 particles per population.
    summary_statistics:
        A function which takes the raw model output as returned by
        any ot the models and calculates the corresponding summary
        statistics. Note that the default value is just the identity
        function. I.e. the model is assumed to already calculate
        the summary statistics. However, in a model selection setting
        it can make sense to have the model produce some kind or raw output
        and then take the same summary statistics function for all the models.
    model_prior:
        A random variable giving the prior weights of the model classes.
        The default is a uniform prior over the model classes,
        ``RV("randint", 0, len(models))``.
    model_perturbation_kernel:
        Kernel which governs with which probability to switch from one
        model to another model for a given sample while generating proposals
        for the subsequent population from the current population.
    transitions:
        A list of :class:`pyabc.transition.Transition` objects
        or a single :class:`pyabc.transition.Transition` in case
        of a single model. Defaults to multivariate normal transitions for
        every model.
    eps:
        Accepts any :class:`pyabc.epsilon.Epsilon` subclass.
        The default is the :class:`pyabc.epsilon.MedianEpsilon` which adapts
        automatically. The object passed here determines how the acceptance
        threshold scheduling is performed.
    sampler:
        In some cases, a mapper implementation will require initialization
        to run properly, e.g. database connection, grid setup, etc..
        The sampler is an object that encapsulates this information.
        The default sampler :class:`pyabc.sampler.MulticoreEvalParallelSampler`
        will parallelize across the cores of a single
        machine only.
    acceptor:
        Takes a distance function, summary statistics and an epsilon threshold
        to decide about acceptance of a particle. Argument accepts any subclass
        of :class:`pyabc.acceptor.Acceptor`, or a function convertible to an
        acceptor. Defaults to a :class:`pyabc.acceptor.UniformAcceptor`.
    stop_if_only_single_model_alive:
        Defaults to False. Set this to true if you want to stop ABCSMC
        automatically as soon as only a single model has survived.
    max_nr_recorded_particles:
        Defaults to inf. Set this to the maximum number of accepted and
        rejected particles that methods like the AdaptivePNormDistance
        function use to update themselves each iteration.


    .. [#tonistumpf] Toni, Tina, and Michael P. H. Stumpf.
        “Simulation-Based Model Selection for Dynamical Systems in Systems and
        Population Biology”.
        Bioinformatics 26, no. 1, 104–10, 2010.
        https://doi.org/10.1093/bioinformatics/btp619
    """

    def __init__(
        self,
        models: Union[List[Model], Model, Callable],
        parameter_priors: Union[List[Distribution], Distribution, Callable],
        distance_function: Union[Distance, Callable] = None,
        population_size: Union[PopulationStrategy, int] = 100,
        summary_statistics: Callable[[model_output], dict] = identity,
        model_prior: RV = None,
        model_perturbation_kernel: ModelPerturbationKernel = None,
        transitions: Union[List[Transition], Transition] = None,
        eps: Epsilon = None,
        sampler: Sampler = None,
        acceptor: Acceptor = None,
        stop_if_only_single_model_alive: bool = False,
        max_nr_recorded_particles: int = np.inf,
    ):
        if not isinstance(models, list):
            models = [models]
        models = list(map(FunctionModel.to_model, models))
        self.models = models

        if not isinstance(parameter_priors, list):
            parameter_priors = [parameter_priors]
        self.parameter_priors = parameter_priors

        # sanity checks
        if len(self.models) != len(self.parameter_priors):
            raise AssertionError(
                "Number models and number parameter priors have to agree."
            )

        if distance_function is None:
            distance_function = PNormDistance()
        self.distance_function = FunctionDistance.to_distance(
            distance_function,
        )

        self.summary_statistics = summary_statistics

        if model_prior is None:
            model_prior = RV("randint", 0, len(self.models))
        self.model_prior = model_prior

        if model_perturbation_kernel is None:
            model_perturbation_kernel = ModelPerturbationKernel(
                len(self.models), probability_to_stay=0.7
            )
        self.model_perturbation_kernel = model_perturbation_kernel

        if transitions is None:
            transitions = [MultivariateNormalTransition() for _ in self.models]
        if not isinstance(transitions, list):
            transitions = [transitions]
        self.transitions: List[Transition] = transitions

        if eps is None:
            eps = MedianEpsilon(median_multiplier=1)
        self.eps = eps

        if isinstance(population_size, int):
            population_size = ConstantPopulationSize(population_size)
        self.population_size = population_size

        if sampler is None:
            sampler = DefaultSampler()
        self.sampler = sampler

        if acceptor is None:
            acceptor = UniformAcceptor()
        self.acceptor = FunctionAcceptor.to_acceptor(acceptor)

        self.stop_if_only_single_model_alive = stop_if_only_single_model_alive
        self.max_nr_recorded_particles = max_nr_recorded_particles

        # will be set later
        self.x_0 = None
        self.history = None
        self._initial_population = None
        self.minimum_epsilon = None
        self.max_nr_populations = None
        self.min_acceptance_rate = None
        self.max_t = None
        self.max_total_nr_simulations = None
        self.max_walltime = None
        self.min_eps_diff = None

        self.init_walltime = None
        self.analysis_id = None

        self._sanity_check()

    def _sanity_check(self):
        # check stochastic setting
        stochastics = [
            isinstance(self.acceptor, StochasticAcceptor),
            isinstance(self.eps, TemperatureBase),
            isinstance(self.distance_function, StochasticKernel),
        ]
        # check if usage is consistent
        if not all(stochastics) and any(stochastics):
            raise ValueError(
                "Please only use acceptor.StochasticAcceptor, "
                "epsilon.TemperatureBase and distance.StochasticKernel "
                "together."
            )

        # check sampler
        self.sampler.check_analysis_variables(
            distance_function=self.distance_function,
            eps=self.eps,
            acceptor=self.acceptor,
        )

    def __getstate__(self):
        state_red_dict = self.__dict__.copy()
        del state_red_dict['sampler']
        return state_red_dict

    def new(
        self,
        db: str,
        observed_sum_stat: dict = None,
        *,
        gt_model: int = None,
        gt_par: dict = None,
        meta_info=None,
    ) -> History:
        """
        Make a new ABCSMC run.

        Parameters
        ----------

        db: str
            Has to be a valid SQLAlchemy database identifier.
            This indicates the database to be used (and created if necessary
            and possible) for the ABC-SMC run.

            To use an in-memory database pass "sqlite://".
            Note that in-memory databases are only available on the master
            mode. If workers are started on different nodes they won't be
            able to access the database. This should not be a problem
            in most scenarios. The in-memory option is mainly useful for
            benchmarking (and maybe) for testing.
        observed_sum_stat : dict, optional
            This is the really important parameter here. It is of the
            form ``{'statistic_1': val_1, 'statistic_2': val_2, ... }``.

            The dictionary provided here represents the measured data.
            Particle during ABCSMC sampling are compared against the
            summary statistics provided here.

            The summary statistics' values can be integers, floats,
            strings and everything which is a numpy array or can be
            converted to one (e.g. lists).
            In addition, pandas.DataFrames can also be used as summary
            statistics.
            **Note that storage of pandas DataFrames in pyABC's database
            is still considered experimental.**

            This parameter is optional, as the distance function might
            implement comparison to the observed data on its own.
            Not giving this parameter is equivalent to passing an empty
            dictionary ``{}``.
        gt_model: int, optional
            This is only meta data stored to the database, but not actually
            used for the ABCSMC algorithm. If you want to predict your ABCSMC
            procedure against synthetic samples, you can use
            this parameter to indicate the ground truth model number.
            This helps with further analysis. If you use actually measured data
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

        Returns
        -------
        history: History
            The history, with set history.id, which is the id under which the
            generated ABCSMC run entry in the database can be identified.
        """
        # record observed summary statistics
        if observed_sum_stat is None:
            observed_sum_stat = {}
        self.x_0 = observed_sum_stat

        # initialize history object
        self.history = History(db)

        if gt_par is None:
            gt_par = {}

        # save configuration data to database
        model_names = [model.name for model in self.models]
        self.history.store_initial_data(
            ground_truth_model=gt_model,
            options=meta_info,
            observed_summary_statistics=observed_sum_stat,
            ground_truth_parameter=gt_par,
            model_names=model_names,
            distance_function_json_str=self.distance_function.to_json(),
            eps_function_json_str=self.eps.to_json(),
            population_strategy_json_str=self.population_size.to_json(),
            start_time=datetime.now(),
        )

        # return history
        # contains id generated in store_initial_data
        return self.history

    def load(
        self,
        db: str,
        abc_id: int = 1,
        observed_sum_stat: dict = None,
    ) -> History:
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
        observed_sum_stat: dict, optional
            The observed summary statistics. This field should be used only if
            the summary statistics cannot be reproduced exactly from the
            database (in particular when they are no numpy or pandas objects,
            e.g. when they were generated in R). If None, then the summary
            statistics are read from the history.
        """
        self.history = History(db)
        self.history.id = abc_id

        # extract observed sum stats from input or history
        if observed_sum_stat is None:
            observed_sum_stat = self.history.observed_sum_stat()
        self.x_0 = observed_sum_stat

        # just return the history
        return self.history

    def _initialize_dist_eps_acc(self, t: int):
        """
        Called once at the start of run(). This function either, if available,
        takes the last population from the history, or generates a
        sample population from the prior. Then, it calls the initialize()
        functions of the distance, epsilon, and acceptor.

        Note that a calibration sample is only taken if required by any of
        the tools.

        Parameters
        ----------
        t: int
            Time point for which to initialize (i.e. the time point at which
            to do the first population). Usually 0 or history.max_t + 1.
        """

        def get_initial_sample():
            """Create sample from population, with only accepted particles."""
            population = self._get_initial_population(t - 1)
            return Sample.from_population(population)

        def _get_initial_population_with_distances():
            population = self._get_initial_population(t - 1)

            def distance_to_ground_truth(x, par):
                return self.distance_function(x, self.x_0, t, par)

            population.update_distances(distance_to_ground_truth)
            return population

        def get_initial_weighted_distances():
            population = _get_initial_population_with_distances()
            weighted_distances = population.get_weighted_distances()
            return weighted_distances

        # initialize dist, eps, acc (order important)
        self.distance_function.initialize(
            t=t,
            get_sample=get_initial_sample,
            x_0=self.x_0,
            total_sims=self.history.total_nr_simulations,
        )
        self.acceptor.initialize(
            t=t,
            get_weighted_distances=get_initial_weighted_distances,
            distance_function=self.distance_function,
            x_0=self.x_0,
        )

        def get_initial_records():
            population = _get_initial_population_with_distances()
            records = []
            for particle in population.particles:
                # we use dummy densities here, since only the quotient
                #  is of interest
                records.append(
                    {
                        'distance': particle.distance,
                        'transition_pd_prev': 1.0,
                        'transition_pd': 1.0,
                        'accepted': True,
                    }
                )
            return records

        self.eps.initialize(
            t=t,
            get_weighted_distances=get_initial_weighted_distances,
            get_all_records=get_initial_records,
            max_nr_populations=self.max_nr_populations,
            acceptor_config=self.acceptor.get_epsilon_config(t),
        )

    def _get_initial_population(self, t: int) -> (List[float], List[dict]):
        """
        Get initial samples, either from the last population stored in history,
        or via sampling sum stats from the prior. This can be used to calibrate
        the distance function or the epsilon.

        The history must have been initialized already. This function fills the
        private property _initial_population.

        .. warning::
            The sample is cached. Thus, the function can be called repeatedly
            without further computational overhead.

        Parameters
        ----------
        t: The time for which to draw samples, i.e. one before the target time.
        """
        if self._initial_population is None:
            if self.history.n_populations > 0:
                # extract latest population from database
                population = self.history.get_population(t=t)
            else:
                # sample
                population = self._sample_from_prior(t)
            self._initial_population = population

        return self._initial_population

    def _create_simulate_from_prior_function(self):
        """
        Similar to _create_simulate_function, apart here we sample from the
        prior and accept all.
        """
        return create_simulate_from_prior_function(
            model_prior=self.model_prior,
            parameter_priors=self.parameter_priors,
            models=self.models,
            summary_statistics=self.summary_statistics,
        )

    def _sample_from_prior(self, t: int) -> Population:
        """
        Only sample from prior and return results without changing
        the history of the distance function or the epsilon.
        """
        # create simulate function
        simulate_one = self._create_simulate_from_prior_function()

        logger.info(f"Calibration sample t = {t}.")

        # call sampler
        sample = self.sampler.sample_until_n_accepted(
            n=self.population_size(-1),
            simulate_one=simulate_one,
            t=t,
            max_eval=np.inf,
            all_accepted=True,
            ana_vars=self._vars(t=t),
        )

        # normalize accepted population weight to 1
        sample.normalize_weights()

        # extract accepted population
        population = sample.get_accepted_population()

        # update information saved in history about calibration
        self.history.update_after_calibration(
            nr_samples=self.sampler.nr_evaluations_, end_time=datetime.now()
        )

        return population

    def _create_simulate_function(self, t: int):
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
        return create_simulate_function(
            t=t,
            model_probabilities=self.history.get_model_probabilities(t - 1),
            model_perturbation_kernel=self.model_perturbation_kernel,
            transitions=self.transitions,
            model_prior=self.model_prior,
            parameter_priors=self.parameter_priors,
            models=self.models,
            summary_statistics=self.summary_statistics,
            x_0=self.x_0,
            distance_function=self.distance_function,
            eps=self.eps,
            acceptor=self.acceptor,
        )

    def _create_transition_pdf(self, t: int, transitions):
        """Create transition probability density function for time `t>=0`."""
        if t == 0:
            return create_prior_pdf(
                model_prior=self.model_prior,
                parameter_priors=self.parameter_priors,
            )

        return create_transition_pdf(
            transitions=transitions,
            model_probabilities=self.history.get_model_probabilities(t - 1),
            model_perturbation_kernel=self.model_perturbation_kernel,
        )

    @run_cleanup
    def run(
        self,
        minimum_epsilon: float = None,
        max_nr_populations: int = np.inf,
        min_acceptance_rate: float = 0.0,
        max_total_nr_simulations: int = np.inf,
        max_walltime: timedelta = None,
        min_eps_diff: float = 0.0,
    ) -> History:
        """
        Run the ABCSMC model selection until either of the stopping
        criteria is met.

        Parameters
        ----------
        minimum_epsilon:
            Stop if epsilon is smaller than minimum epsilon specified here.
            Defaults in general to 0.0, and to 1.0 for a Temperature epsilon.
        max_nr_populations:
            Maximum allowed number of populations.
            Stop if this number is reached.
        min_acceptance_rate:
            Minimal allowed acceptance rate. Sampling stops if a population
            has a lower rate.
        max_total_nr_simulations:
            Maximum allowed total number of evaluations.
        max_walltime:
            Maximum allowed walltime since start of the run() method,
            e.g.

            >>> from datetime import timedelta
            >>> max_walltime = timedelta(hours=2, minutes=30)

        min_eps_diff:
            Minimum epsilon difference in two sequential generations.

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
        t0: int = self.initialize_components_before_run(
            minimum_epsilon=minimum_epsilon,
            max_nr_populations=max_nr_populations,
            min_acceptance_rate=min_acceptance_rate,
            max_total_nr_simulations=max_total_nr_simulations,
            max_walltime=max_walltime,
            min_eps_diff=min_eps_diff,
        )

        # run loop over time points
        t = t0
        while True:
            # perform one generation
            ret = self.run_generation(t=t)

            # check whether to discontinue
            if not ret["successful"] or self.check_terminate(
                t=t,
                acceptance_rate=ret["acceptance_rate"],
            ):
                break

            # increment t
            t += 1

        # return used history object
        return self.history

    def initialize_components_before_run(
        self,
        minimum_epsilon: float,
        max_nr_populations: int,
        min_acceptance_rate: float,
        max_total_nr_simulations: int,
        max_walltime: timedelta,
        min_eps_diff: float,
    ) -> int:
        """Initialize everything before starting a run.

        In particular sets variables corresponding to arguments, and
        performs sampling based initialization for e.g. distance and epsilon,
        if these are adaptive.

        All parameters are as for `run()`.

        Returns
        -------
        t0: The initial time point from which to start the next generation.
        """
        # handle arguments
        if minimum_epsilon is None:
            if isinstance(self.eps, TemperatureBase):
                minimum_epsilon = 1.0
            else:
                minimum_epsilon = 0.0
        self.minimum_epsilon = minimum_epsilon

        self.max_nr_populations = max_nr_populations
        self.min_acceptance_rate = min_acceptance_rate
        self.max_total_nr_simulations = max_total_nr_simulations
        self.max_walltime = max_walltime
        self.min_eps_diff = min_eps_diff

        # for recording the overall time
        self.init_walltime = datetime.now()

        # initial time index
        t0 = self.history.max_t + 1
        # last time point index
        self.max_t = t0 + max_nr_populations - 1

        # assign an analysis id
        self.analysis_id = create_analysis_id()
        # let the sampler know
        self.sampler.set_analysis_id(self.analysis_id)

        # initialize transitions
        self._fit_transitions(t0)
        # initialize population size
        self._adapt_population_size(t0)
        # sample from prior to calibrate distance, epsilon, and acceptor
        self._initialize_dist_eps_acc(t0)

        # configure recording of rejected particles
        self.distance_function.configure_sampler(self.sampler)
        # TODO update configuration in each iteration
        self.eps.configure_sampler(self.sampler)

        return t0

    def run_generation(
        self,
        t: int,
    ) -> dict:
        """Run a single generation.

        Parameters
        ----------
        t: Generation time index to run for.

        Returns
        -------
        ret:
            Dictionary with entries "successful" indicating whether the
            generation terminated successfully,
            and potentially "acceptance_rate".
        """
        # get epsilon for generation t
        current_eps = self.eps(t)
        if current_eps is None or np.isnan(current_eps):
            raise ValueError(
                f"The epsilon threshold {current_eps} is invalid."
            )
        logger.info(f"t: {t}, eps: {current_eps:.8e}.")

        # create simulate function
        simulate_one = self._create_simulate_function(t)

        # population size and maximum number of evaluations
        pop_size = self.population_size(t)
        max_eval = (
            np.inf
            if self.min_acceptance_rate == 0.0
            else pop_size / self.min_acceptance_rate
        )

        # perform the sampling
        logger.debug(f"Submitting population {t}.")
        sample = self.sampler.sample_until_n_accepted(
            n=pop_size,
            simulate_one=simulate_one,
            t=t,
            max_eval=max_eval,
            ana_vars=self._vars(t=t),
        )

        # check sample health
        if not sample.ok:
            logger.info("Stopping: sample not ok.")
            return {
                "successful": False,
            }

        # normalize accepted population weight to 1
        sample.normalize_weights()

        # retrieve accepted population
        population = sample.get_accepted_population()
        logger.debug(f"Population {t} done.")

        # save to database
        n_sim = self.sampler.nr_evaluations_
        model_names = [model.name for model in self.models]
        self.history.append_population(
            t, current_eps, population, n_sim, model_names
        )
        logger.debug(
            f"Total samples up to t = {t}: "
            f"{self.history.total_nr_simulations}."
        )

        # acceptance rate and ess
        pop_size = len(population)
        acceptance_rate = pop_size / n_sim
        ess = effective_sample_size(population.get_weighted_distances()['w'])
        logger.info(
            f"Accepted: {pop_size} / {n_sim} = "
            f"{acceptance_rate:.4e}, ESS: {ess:.4e}."
        )

        # prepare next iteration
        self._prepare_next_iteration(
            t=t + 1,
            sample=sample,
            population=population,
            acceptance_rate=acceptance_rate,
        )

        return {
            "successful": True,
            "acceptance_rate": acceptance_rate,
        }

    def check_terminate(
        self,
        t: int,
        acceptance_rate: float,
    ) -> bool:
        """Check whether any termination criterion is met.

        Parameters
        ----------
        t: Current time point.
        acceptance_rate: Acceptance rate in current generation.

        Returns
        -------
        terminate: Whether to terminate (True) or not (False).
        """
        # check termination criteria
        if termination_criteria_fulfilled(
            current_eps=self.eps(t=t),
            min_eps=self.minimum_epsilon,
            prev_eps=eps_from_hist(history=self.history, t=t - 1),
            min_eps_diff=self.min_eps_diff,
            stop_if_single_model_alive=self.stop_if_only_single_model_alive,
            # noqa: E251
            nr_of_models_alive=self.history.nr_of_models_alive(),
            acceptance_rate=acceptance_rate,
            min_acceptance_rate=self.min_acceptance_rate,
            total_nr_simulations=self.history.total_nr_simulations,
            max_total_nr_simulations=self.max_total_nr_simulations,
            walltime=datetime.now() - self.init_walltime,
            max_walltime=self.max_walltime,
            t=t,
            max_t=self.max_t,
        ):
            return True
        return False

    def _prepare_next_iteration(
        self,
        t: int,
        sample: Sample,
        population: Population,
        acceptance_rate: float,
    ):
        """Update actors for the upcoming iteration.

        Be aware: The current (finished) iteration is t-1, the next t.

        Parameters
        ----------
        t: int
            The upcoming iteration time index to prepare for.
        sample: pyabc.Sample
            The current iteration's sample object.
        population: pyabc.Population
            The current iteration's population object.
        acceptance_rate: float
            The current iteration's acceptance rate.
        """
        # make a copy
        prev_transitions = copy.deepcopy(self.transitions)

        # update transitions
        self._fit_transitions(t)

        # update population size
        self._adapt_population_size(t)

        def get_sample():
            return sample

        # update distance
        df_updated = self.distance_function.update(
            t=t,
            get_sample=get_sample,
            total_sims=self.history.total_nr_simulations,
        )

        # compute distances with the new distance measure
        def get_weighted_distances():
            if df_updated:

                def distance_to_ground_truth(x, par):
                    return self.distance_function(x, self.x_0, t, par)

                population.update_distances(distance_to_ground_truth)
            return population.get_weighted_distances()

        # update acceptor
        self.acceptor.update(
            t=t,
            get_weighted_distances=get_weighted_distances,
            prev_temp=self.eps(t - 1),
            acceptance_rate=acceptance_rate,
        )

        def get_all_records():
            recorded_particles = sample.all_particles

            # create list of all records
            records = []
            # get transition functions
            transition_pdf_prev = self._create_transition_pdf(
                t - 1, prev_transitions
            )
            transition_pdf = self._create_transition_pdf(t, self.transitions)

            # iterate over all particles
            for particle in recorded_particles:
                # evaluate previous and current transition density
                transition_pd_prev = transition_pdf_prev(
                    particle.m, particle.parameter
                )
                transition_pd = transition_pdf(particle.m, particle.parameter)
                records.append(
                    {
                        'distance': particle.distance,
                        'transition_pd_prev': transition_pd_prev,
                        'transition_pd': transition_pd,
                        'accepted': particle.accepted,
                    }
                )
            return records

        # update epsilon
        self.eps.update(
            t=t,
            get_weighted_distances=get_weighted_distances,
            get_all_records=get_all_records,
            acceptance_rate=acceptance_rate,
            acceptor_config=self.acceptor.get_epsilon_config(t),
        )

    def _adapt_population_size(self, t):
        """
        Adapt population size based on the employed population strategy.

        Parameters
        ----------
        t: int
            Time for which to adapt the population size.
        """
        if t == 0:  # we need a particle population to do the fitting
            return

        # model probabilities
        w = self.history.get_model_probabilities(self.history.max_t)[
            "p"
        ].values

        # make a copy in case the population strategy messes with
        # the transitions
        # WARNING: the deepcopy also copies the random states of scipy.stats
        # distributions
        copied_transitions = copy.deepcopy(self.transitions)

        # update the population size
        self.population_size.update(
            transitions=copied_transitions, model_weights=w, t=t
        )

    def _fit_transitions(self, t):
        """
        Fit the density estimator.

        Parameters
        ----------
        t: int
            Time for which to update the kernel density estimator.
        """
        if t == 0:  # we need a particle population to do the fitting
            return

        for m in self.history.alive_models(t - 1):
            particles, w = self.history.get_distribution(m, t - 1)
            self.transitions[m].fit(particles, w)

    def _vars(self, t: int) -> AnalysisVars:
        """Create a dictionary of analysis variables of interest.

        These variables are passed to the sampler, as some need to create
        simulation settings themselves.
        """

        return AnalysisVars(
            model_prior=self.model_prior,
            parameter_priors=self.parameter_priors,
            model_perturbation_kernel=self.model_perturbation_kernel,
            transitions=self.transitions,
            models=self.models,
            summary_statistics=self.summary_statistics,
            x_0=self.x_0,
            distance_function=self.distance_function,
            eps=self.eps,
            acceptor=self.acceptor,
            min_acceptance_rate=self.min_acceptance_rate,
            min_eps=self.minimum_epsilon,
            stop_if_single_model_alive=self.stop_if_only_single_model_alive,
            max_t=self.max_t,
            max_total_nr_simulations=self.max_total_nr_simulations,
            prev_total_nr_simulations=self.history.total_nr_simulations,
            max_walltime=self.max_walltime,
            init_walltime=self.init_walltime,
            min_eps_diff=self.min_eps_diff,
            prev_eps=eps_from_hist(history=self.history, t=t - 1),
        )
