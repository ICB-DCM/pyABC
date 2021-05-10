"""Redis based sampler base class and dynamic scheduling samplers."""

import numpy as np
from time import sleep
from datetime import datetime
import cloudpickle as pickle
import copy
import logging
from redis import StrictRedis
from typing import Callable, List, Tuple
from jabbar import jabbar

from ...util import (
    AnalysisVars, create_simulate_function, evaluate_preliminary_particle,
    termination_criteria_fulfilled)
from ...distance import Distance
from ...epsilon import Epsilon
from ...acceptor import Acceptor
from ...sampler import Sampler, Sample
from ...weighted_statistics import effective_sample_size
from .cmd import (
    SSA, N_EVAL, N_ACC, N_REQ, N_FAIL, N_LOOKAHEAD_EVAL, ALL_ACCEPTED,
    N_WORKER, QUEUE, MSG, START, MODE, DYNAMIC, SLEEP_TIME, BATCH_SIZE,
    IS_LOOK_AHEAD, ANALYSIS_ID, GENERATION, MAX_N_EVAL_LOOK_AHEAD, ACTIVE_SET,
    idfy)
from .util import get_active_set
from .redis_logging import RedisSamplerLogger

logger = logging.getLogger("ABC.Sampler")


class RedisSamplerBase(Sampler):
    """Abstract base class for redis based samplers.

    Parameters
    ----------
    host:
        IP address or name of the Redis server.
        Default is "localhost".
    port:
        Port of the Redis server.
        Default is 6379.
    password:
        Password for a protected server. Default is None (no protection).
    log_file:
        A file for a dedicated sampler history. Updated in each iteration.
        This log file is complementary to the logging realized via the
        logging module.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str = None,
        log_file: str = None,
    ):
        super().__init__()
        logger.debug(
            f"Redis sampler: host={host} port={port}")
        # handles the connection to the redis-server
        self.redis: StrictRedis = StrictRedis(
            host=host, port=port, password=password)
        self.logger = RedisSamplerLogger(log_file)

    def n_worker(self) -> int:
        """
        Get the number of connected workers.

        Returns
        -------

        Number of workers connected.
        """
        return self.redis.pubsub_numsub(MSG)[0][-1]

    def set_analysis_id(self, analysis_id: str):
        """Set the analysis id and make sure the server is available."""
        super().set_analysis_id(analysis_id)
        if self.redis.get(ANALYSIS_ID):
            raise AssertionError(
                "The server seems busy with an analysis already")
        self.redis.set(ANALYSIS_ID, analysis_id)

    def sample_until_n_accepted(
        self,
        n: int,
        simulate_one: Callable,
        t: int,
        *,
        max_eval: int = np.inf,
        all_accepted: bool = False,
        ana_vars: AnalysisVars = None,
    ) -> Sample:
        raise NotImplementedError()

    def stop(self):
        """Stop potentially still ongoing sampling."""
        # delete ids specifying the current analysis
        self.redis.delete(ANALYSIS_ID)
        self.redis.delete(idfy(GENERATION, self.analysis_id))
        # note: the other ana_id-t-specific variables are not deleted, as these
        #  do not hurt anyway and could potentially make the workers fail


class RedisEvalParallelSampler(RedisSamplerBase):
    """Redis based dynamic scheduling low latency sampler.

    This sampler is well-performing in distributed environments.
    It is usually faster than the
    :class:`pyabc.sampler.DaskDistributedSampler` for short model evaluation
    runtimes. The longer the model evaluation times, the less the advantage
    becomes. It requires a running Redis server as broker.

    This sampler requires workers to be started via the command
    ``abc-redis-worker``.
    An example call might look like
    ``abc-redis-worker --host=123.456.789.123 --runtime=2h``
    to connect to a Redis server on IP ``123.456.789.123`` and to terminate
    the worker after finishing the first population which ends after 2 hours
    since worker start. So the actual runtime might be longer than 2h.
    See ``abc-redis-worker --help`` for its options.

    Use the command ``abc-redis-manager`` to retrieve info on and stop the
    running workers.

    Start as many workers as you wish. Workers can be dynamically added
    during the ABC run.

    Currently, a server (specified via host and port) can only meaningfully
    handle one ABCSMC analysis at a time.

    Parameters
    ----------
    host:
        IP address or name of the Redis server.
        Default is "localhost".
    port:
        Port of the Redis server.
        Default is 6379.
    password:
        Password for a protected server. Default is None (no protection).
    batch_size:
        Number of model evaluations the workers perform before contacting
        the REDIS server. Defaults to 1. Increase this value if model
        evaluation times are short or the number of workers is large
        to reduce communication overhead.
    look_ahead:
        Whether to start sampling for the next generation already with
        preliminary results although the current generation has not completely
        finished yet. This increases parallel efficiency, but can lead to
        a higher Monte-Carlo error.
    look_ahead_delay_evaluation:
        In look-ahead mode, acceptance can be delayed until the final
        acceptance criteria for generation t have been decided. This is
        mandatory if the routine has adaptive components (distance, epsilon,
        ...) besides the transition kernel. If not needed, enabling it may
        lead to a worse performance, especially if evaluation is costly
        compared to simulation, because evaluation happens sequentially on the
        main thread.
        Only effective if `look_ahead=True`.
    max_n_eval_look_ahead_factor:
        In delayed evaluation, only this factor times the previous number of
        samples are generated, afterwards the workers wait.
        Does not apply if evaluation is not delayed.
        This allows to perform a reasonable number of evaluations only, as
        for short-running models the number of evaluations can otherwise
        explode unnecessarily.
    wait_for_all_samples:
        Whether to wait for all simulations in an iteration to finish.
        If not, then the sampler only waits for all simulations that were
        started prior to the last started particle of the first `n`
        acceptances.
    log_file:
        A file for a dedicated sampler history. Updated in each iteration.
        This log file is complementary to the logging realized via the
        logging module.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        password: str = None,
        batch_size: int = 1,
        look_ahead: bool = False,
        look_ahead_delay_evaluation: bool = True,
        max_n_eval_look_ahead_factor: float = 10.,
        wait_for_all_samples: bool = False,
        log_file: str = None,
    ):
        super().__init__(
            host=host, port=port, password=password, log_file=log_file)
        self.batch_size: int = batch_size
        self.look_ahead: bool = look_ahead
        self.look_ahead_delay_evaluation: bool = look_ahead_delay_evaluation
        self.max_n_eval_look_ahead_factor: float = max_n_eval_look_ahead_factor
        self.wait_for_all_samples: bool = wait_for_all_samples

    def sample_until_n_accepted(
            self, n, simulate_one, t, *,
            max_eval=np.inf, all_accepted=False, ana_vars=None):
        # get the analysis id
        ana_id = self.analysis_id

        def get_int(var: str):
            """Convenience function to read an int variable."""
            return int(self.redis.get(idfy(var, ana_id, t)).decode())

        if self.generation_t_was_started(t):
            # update the SSA function
            self.redis.set(
                idfy(SSA, ana_id, t),
                pickle.dumps((simulate_one, self.sample_factory)))
            # update the required population size
            self.redis.set(idfy(N_REQ, ana_id, t), n)
            # let the workers know they should update their ssa
            self.redis.set(idfy(IS_LOOK_AHEAD, ana_id, t), int(False))
            # it can happen that the population size increased, but the workers
            #  believe they are done already
            if get_int(N_WORKER) == 0 and get_int(N_ACC) < get_int(N_REQ):
                # send the start signal again
                self.redis.publish(MSG, START)
        else:
            # set up all variables for the new generation
            self.start_generation_t(
                n=n, t=t, simulate_one=simulate_one, all_accepted=all_accepted,
                is_look_ahead=False)

        # for the results
        id_results = []
        # reset logging counters
        self.logger.reset_counters()

        # wait until n acceptances
        with jabbar(total=n, enable=self.show_progress, keep=False) as bar:
            while len(id_results) < n:
                # pop result from queue, block until one is available
                dump = self.redis.blpop(idfy(QUEUE, ana_id, t))[1]
                # extract pickled object
                sample_with_id = pickle.loads(dump)

                # check whether the sample is really acceptable
                sample_with_id, any_particle_accepted = \
                    post_check_acceptance(
                        sample_with_id, ana_id=ana_id, t=t, redis=self.redis,
                        ana_vars=ana_vars, logger=self.logger)

                if any_particle_accepted:
                    # append to collected results
                    id_results.append(sample_with_id)
                    bar.update(len(id_results))

        # log active set
        _log_active_set(
            redis=self.redis, ana_id=ana_id, t=t, id_results=id_results,
            batch_size=self.batch_size)

        # maybe head-start the next generation already
        self.maybe_start_next_generation(
            t=t, n=n, id_results=id_results, all_accepted=all_accepted,
            ana_vars=ana_vars)

        # wait until all relevant simulations done
        if self.wait_for_all_samples:
            while get_int(N_WORKER) > 0:
                sleep(SLEEP_TIME)
        else:
            max_ix = max(id_result[0] for id_result in id_results)
            while (
                # check whether any active evaluation was started earlier
                any(ix <= max_ix for ix in get_active_set(
                    redis=self.redis, ana_id=ana_id, t=t))
                # also stop if no worker is active, useful for server resets
                and get_int(N_WORKER) > 0
            ):
                sleep(SLEEP_TIME)

        # collect all remaining results in queue at this point
        while self.redis.llen(idfy(QUEUE, ana_id, t)) > 0:
            # pop result from queue, block until one is available
            dump = self.redis.blpop(idfy(QUEUE, ana_id, t))[1]
            # extract pickled object
            sample_with_id = pickle.loads(dump)

            # check whether the sample is really acceptable
            sample_with_id, any_particle_accepted = \
                post_check_acceptance(
                    sample_with_id, ana_id=ana_id, t=t, redis=self.redis,
                    ana_vars=ana_vars, logger=self.logger)

            if any_particle_accepted:
                # append to collected results
                id_results.append(sample_with_id)

        # set total number of evaluations
        self.nr_evaluations_ = get_int(N_EVAL)
        n_lookahead_eval = get_int(N_LOOKAHEAD_EVAL)

        # remove all time-specific variables if no more active workers,
        #  also for previous generations
        for _t in range(-1, t+1):
            n_worker_b = self.redis.get(idfy(N_WORKER, ana_id, _t))
            if n_worker_b is not None and int(n_worker_b.decode()) == 0:
                self.clear_generation_t(t=_t)

        # create a single sample result, with start time correction
        sample = self.create_sample(id_results, n)

        # logging
        self.logger.add_row(
            t=t, n_evaluated=self.nr_evaluations_,
            n_lookahead=n_lookahead_eval)
        self.logger.write()

        # weight samples correctly
        sample = self_normalize_within_subpopulations(sample, n)

        return sample

    def start_generation_t(
        self,
        n: int,
        t: int,
        simulate_one: Callable,
        all_accepted: bool,
        is_look_ahead: bool,
        max_n_eval_look_ahead: float = np.inf,
    ) -> None:
        """Start generation `t`."""
        ana_id = self.analysis_id

        # write initial values to pipeline
        (self.redis.pipeline()
         # initialize variables for time t
         .set(idfy(SSA, ana_id, t),
              pickle.dumps((simulate_one, self.sample_factory)))
         .set(idfy(N_EVAL, ana_id, t), 0)
         .set(idfy(N_ACC, ana_id, t), 0)
         .set(idfy(N_REQ, ana_id, t), n)
         .set(idfy(N_FAIL, ana_id, t), 0)
         .set(idfy(N_LOOKAHEAD_EVAL, ana_id, t), 0)
         # encode as int
         .set(idfy(ALL_ACCEPTED, ana_id, t), int(all_accepted))
         .set(idfy(N_WORKER, ana_id, t), 0)
         .set(idfy(BATCH_SIZE, ana_id, t), self.batch_size)
         # encode as int
         .set(idfy(IS_LOOK_AHEAD, ana_id, t), int(is_look_ahead))
         .set(idfy(MAX_N_EVAL_LOOK_AHEAD, ana_id, t), max_n_eval_look_ahead)
         .set(idfy(MODE, ana_id, t), DYNAMIC)
         .set(idfy(ACTIVE_SET, ana_id, t), pickle.dumps(set()))
         # update the current-generation variable
         .set(idfy(GENERATION, ana_id), t)
         # execute all commands
         .execute())

        # publish start message
        self.redis.publish(MSG, START)

    def generation_t_was_started(self, t: int) -> bool:
        """Check whether generation `t` was started already.

        Parameters
        ----------
        t: The time for which to check.
        """
        # just check any of the variables for time t
        return self.redis.exists(idfy(N_REQ, self.analysis_id, t))

    def clear_generation_t(self, t: int) -> None:
        """Clean up after generation `t` has finished.

        Parameters
        ----------
        t: The time for which to clear.
        """
        ana_id = self.analysis_id
        # delete keys from pipeline
        (self.redis.pipeline()
         .delete(idfy(SSA, ana_id, t))
         .delete(idfy(N_EVAL, ana_id, t))
         .delete(idfy(N_ACC, ana_id, t))
         .delete(idfy(N_REQ, ana_id, t))
         .delete(idfy(N_FAIL, ana_id, t))
         .delete(idfy(N_LOOKAHEAD_EVAL, ana_id, t))
         .delete(idfy(ALL_ACCEPTED, ana_id, t))
         .delete(idfy(N_WORKER, ana_id, t))
         .delete(idfy(BATCH_SIZE, ana_id, t))
         .delete(idfy(IS_LOOK_AHEAD, ana_id, t))
         .delete(idfy(MAX_N_EVAL_LOOK_AHEAD, ana_id, t))
         .delete(idfy(MODE, ana_id, t))
         .delete(idfy(ACTIVE_SET, ana_id, t))
         .delete(idfy(QUEUE, ana_id, t))
         .execute())

    def maybe_start_next_generation(
        self,
        t: int,
        n: int,
        id_results: List,
        all_accepted: bool,
        ana_vars: AnalysisVars,
    ) -> None:
        """Start the next generation already, if that looks reasonable.

        Parameters
        ----------
        t: The current time.
        n: The current population size.
        id_results: The so-far returned samples.
        all_accepted: Whether all particles are accepted.
        ana_vars: Analysis variables.
        """
        # not in a look-ahead mood
        if not self.look_ahead:
            return

        # all accepted indicates the preliminary iteration, where we don't
        #  want to look ahead yet
        if all_accepted:
            return

        # create a result sample
        sample = self.create_sample(id_results, n)
        # copy as we modify the particles
        sample = copy.deepcopy(sample)

        # extract population
        population = sample.get_accepted_population()

        # acceptance rate
        nr_evaluations = int(
            self.redis.get(idfy(N_EVAL, self.analysis_id, t)).decode())
        acceptance_rate = len(population) / nr_evaluations

        # check if any termination criterion (based on the current data)
        #  is likely to be fulfilled after the current generation

        total_nr_simulations = \
            ana_vars.prev_total_nr_simulations + nr_evaluations
        walltime = datetime.now() - ana_vars.init_walltime

        if termination_criteria_fulfilled(
            current_eps=ana_vars.eps(t),
            min_eps=ana_vars.min_eps,
            stop_if_single_model_alive=  # noqa: E251
            ana_vars.stop_if_single_model_alive,
            nr_of_models_alive=population.nr_of_models_alive(),
            acceptance_rate=acceptance_rate,
            min_acceptance_rate=ana_vars.min_acceptance_rate,
            total_nr_simulations=total_nr_simulations,
            max_total_nr_simulations=ana_vars.max_total_nr_simulations,
            walltime=walltime,
            max_walltime=ana_vars.max_walltime,
            t=t, max_t=ana_vars.max_t,
        ):
            # do not head-start a new generation as this is likely not needed
            return

        # create a preliminary simulate_one function
        simulate_one_prel = create_preliminary_simulate_one(
            t=t + 1, population=population,
            delay_evaluation=self.look_ahead_delay_evaluation,
            ana_vars=ana_vars)

        # maximum number of look-ahead evaluations
        if self.look_ahead_delay_evaluation:
            # set maximum evaluations to previous simulations * const
            nr_evaluations_ = int(
                self.redis.get(idfy(N_EVAL, self.analysis_id, t)).decode())
            max_n_eval_look_ahead = \
                nr_evaluations_ * self.max_n_eval_look_ahead_factor
        else:
            # no maximum necessary as samples are directly evaluated
            max_n_eval_look_ahead = np.inf

        # head-start the next generation
        #  all_accepted is most certainly False for t>0
        self.start_generation_t(
            n=n, t=t + 1, simulate_one=simulate_one_prel,
            all_accepted=False, is_look_ahead=True,
            max_n_eval_look_ahead=max_n_eval_look_ahead)

    def create_sample(self, id_results: List[Tuple], n: int) -> Sample:
        """Create a single sample result.
        Order the results by starting point to avoid a bias towards
        short-running simulations (dynamic scheduling).
        """
        # sort
        id_results.sort(key=lambda x: x[0])
        # cut off
        id_results = id_results[:n]

        # extract simulations
        results = [res[1] for res in id_results]

        # create 1 to-be-returned sample from results
        sample = self._create_empty_sample()
        for j in range(n):
            sample += results[j]

        return sample

    def check_analysis_variables(
        self,
        distance_function: Distance,
        eps: Epsilon,
        acceptor: Acceptor,
    ) -> None:
        """"Check analysis variables appropriateness for sampling."""
        if self.look_ahead_delay_evaluation:
            # nothing to be done
            return

        def check_bad(var):
            """Check whether a component is incompatible."""
            # do not check for `requires_calibration()`, because in the first
            #  iteration we do not look ahead
            if var.is_adaptive():
                raise AssertionError(
                    f"{var.__class__.__name__} cannot be used in look-ahead "
                    "mode without delayed acceptance. Consider setting the "
                    "sampler's `look_ahead_delay_evaluation` flag.")

        check_bad(acceptor)
        check_bad(distance_function)
        check_bad(eps)


def create_preliminary_simulate_one(
    t, population, delay_evaluation: bool, ana_vars: AnalysisVars,
) -> Callable:
    """Create a preliminary simulate_one function for generation `t`.

    Based on preliminary results, update transitions, distance function,
    epsilon threshold etc., and return a function that samples parameters,
    simulates data and checks their preliminary acceptance.
    As the actual acceptance criteria may be different, samples generated by
    this function must be checked anew afterwards.

    Parameters
    ----------
    t: The time index for which to create the function (i.e. call with t+1).
    population: The preliminary population.
    delay_evaluation: Whether to delay evaluation.
    ana_vars: The analysis variables.

    Returns
    -------
    simulate_one: The preliminary sampling function.
    """
    model_probabilities = population.get_model_probabilities()

    # create deep copy of the transition function
    transitions = copy.deepcopy(ana_vars.transitions)

    # fit transition
    for m in population.get_alive_models():
        parameters, w = population.get_distribution(m)
        transitions[m].fit(parameters, w)

    return create_simulate_function(
        t=t, model_probabilities=model_probabilities,
        model_perturbation_kernel=ana_vars.model_perturbation_kernel,
        transitions=transitions, model_prior=ana_vars.model_prior,
        parameter_priors=ana_vars.parameter_priors,
        models=ana_vars.models, summary_statistics=ana_vars.summary_statistics,
        x_0=ana_vars.x_0, distance_function=ana_vars.distance_function,
        eps=ana_vars.eps, acceptor=ana_vars.acceptor,
        evaluate=not delay_evaluation, proposal_id=-1,
    )


def post_check_acceptance(
    sample_with_id, ana_id, t, redis, ana_vars,
    logger: RedisSamplerLogger,
) -> Tuple:
    """Check whether the sample is really acceptable.

    This is where evaluation of preliminary samples happens, using the analysis
    variables from the actual generation `t` and the previously simulated data.
    The sample is modified in-place.

    Returns
    -------
    sample_with_id, any_accepted:
        The (maybe post-evaluated) id-sample tuple, and an indicator whether
        any particle in the sample was accepted, s.t. the sample should be
        kept.
    """
    # 0 is relative start time, 1 is the actual sample
    sample = sample_with_id[1]

    # check whether there are preliminary particles
    if not any(particle.preliminary for particle in sample.particles):
        n_accepted = sum(particle.accepted for particle in sample.particles)
        if n_accepted != 1:
            # this should never happen
            raise AssertionError(
                "Expected exactly one accepted particle in sample.")

        # increase general acceptance counter
        logger.n_accepted += 1

        # increase accepted counter if in look-ahead mode
        if sample.is_look_ahead:
            logger.n_lookahead_accepted += 1

        # nothing else to be done
        return sample_with_id, True

    # in preliminary mode, there should only be one particle per sample
    if len(sample.particles) != 1:
        # this should never happen
        raise AssertionError(
            "Expected number of particles in sample: 1. "
            f"Got: {len(sample.particles)}")

    # from here on, we may assume that all particles (#=1) are yet to be judged
    logger.n_preliminary += 1

    # iterate over the 1 particle
    for i_particle, particle in enumerate(sample.particles):
        sample.particles[i_particle] = \
            evaluate_preliminary_particle(particle, t, ana_vars)

        # react to acceptance
        if sample.particles[i_particle].accepted:
            # increase redis shared counter
            redis.incr(idfy(N_ACC, ana_id, t), 1)
            # increase general and lookahead counter
            logger.n_accepted += 1
            logger.n_lookahead_accepted += 1

    return (sample_with_id,
            any(particle.accepted for particle in sample.particles))


def self_normalize_within_subpopulations(sample: Sample, n: int) -> Sample:
    """Applies subpopulation-wise self-normalization of samples, in-place.

    Parameters
    ----------
    sample: The population to be returned by the sampler.
    n: Population size.

    Returns
    -------
    sample: The same, weight-adjusted sample.
    """
    prop_ids = {particle.proposal_id for particle in sample.particles}

    if len(prop_ids) == 1:
        # Nothing to be done, as we only have one proposal, and normalization
        #  is applied later when the population is created
        return sample

    accepted_particles = sample.accepted_particles
    if len(accepted_particles) != n:
        # this should not happen
        raise AssertionError("Unexpected number of acceptances")

    # get particles per proposal
    particles_per_prop = {
        prop_id: [particle for particle in accepted_particles
                  if particle.proposal_id == prop_id]
        for prop_id in prop_ids}

    # normalize weights by ESS_l / sum_l[w_l] for proposal id l
    # this is s.t. sum_i w_{l,i} \propto ESS_l
    normalizations = {}
    for prop_id, particles_for_prop in particles_per_prop.items():
        weights = np.array(
            [particle.weight for particle in particles_for_prop])
        ess = effective_sample_size(weights)
        total_weight = weights.sum()
        normalizations[prop_id] = ess / total_weight

    # normalize every particle (this includes rejected ones, which should not
    #  be necessary, but does not hurt)
    for particle in sample.particles:
        particle.weight *= normalizations[particle.proposal_id]

    return sample


def _log_active_set(
    redis: StrictRedis,
    ana_id: str,
    t: int,
    id_results: List[Tuple],
    batch_size: int,
) -> None:
    """Log the status of active simulations after the first n acceptances."""
    accepted_ids = [id_result[0] for id_result in id_results]
    active_set = get_active_set(redis=redis, ana_id=ana_id, t=t)
    # remove entries that are already accepted (runtime conditions)
    active_set = active_set.difference(accepted_ids)
    earlier = {ix for ix in active_set if max(accepted_ids) > ix}
    logger.debug(
        f"After {len(accepted_ids)} acceptances, "
        f"{len(active_set) * batch_size} simulations busy, "
        f"thereof {len(earlier) * batch_size} earlier.")
