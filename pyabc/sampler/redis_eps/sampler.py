import numpy as np
import pickle
from time import sleep
from datetime import datetime
import cloudpickle
import copy
from redis import StrictRedis
from typing import Callable, List, Tuple
from jabbar import jabbar

from ...util import (
    AnalysisVars, create_simulate_function, termination_criteria_fulfilled)
from ...sampler import Sampler, Sample
from .cmd import (SSA, N_EVAL, N_ACC, N_REQ, N_FAIL, ALL_ACCEPTED,
                  N_WORKER, QUEUE, MSG, START,
                  SLEEP_TIME, BATCH_SIZE, IS_PREL, ANALYSIS_ID, GENERATION,
                  idfy)
from .redis_logging import logger


class RedisEvalParallelSampler(Sampler):
    """Redis based low latency sampler.

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
    """
    def __init__(self,
                 host: str = "localhost",
                 port: int = 6379,
                 password: str = None,
                 batch_size: int = 1,
                 look_ahead: bool = False):
        super().__init__()
        logger.debug(
            f"Redis sampler: host={host} port={port}")
        # handles the connection to the redis-server
        self.redis: StrictRedis = StrictRedis(
            host=host, port=port, password=password)
        self.batch_size: int = batch_size
        self.look_ahead: bool = look_ahead

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
            self, n, simulate_one, t, *,
            max_eval=np.inf, all_accepted=False, ana_vars=None):
        # get the analysis id
        ana_id = self.analysis_id

        if self.generation_t_was_started(t):
            # update the SSA function
            self.redis.set(
                idfy(SSA, ana_id, t),
                cloudpickle.dumps((simulate_one, self.sample_factory)))
            # let the workers know they should update their ssa
            self.redis.set(idfy(IS_PREL, ana_id, t), int(False))
        else:
            # set up all variables for the new generation
            self.start_generation_t(
                n=n, t=t, simulate_one=simulate_one, all_accepted=all_accepted,
                is_prel=False)

        id_results = []

        # wait until n acceptances
        with jabbar(total=n, enable=self.show_progress, keep=False) as bar:
            while len(id_results) < n:
                # pop result from queue, block until one is available
                dump = self.redis.blpop(idfy(QUEUE, ana_id, t))[1]
                # extract pickled object
                particle_with_id = pickle.loads(dump)
                # TODO check whether the acceptance criterion changed
                # append to collected results
                id_results.append(particle_with_id)
                bar.update(len(id_results))

        # maybe head-start the next generation already
        self.maybe_start_next_generation(
            t=t, n=n, id_results=id_results, ana_vars=ana_vars)

        # wait until all workers done
        while int(self.redis.get(idfy(N_WORKER, ana_id, t)).decode()) > 0:
            sleep(SLEEP_TIME)

        # make sure all results are collected
        while self.redis.llen(idfy(QUEUE, ana_id, t)) > 0:
            id_results.append(
                pickle.loads(self.redis.blpop(idfy(QUEUE, ana_id, t))[1]))

        # set total number of evaluations
        self.nr_evaluations_ = int(
            self.redis.get(idfy(N_EVAL, ana_id, t)).decode())

        # remove all time-specific variables
        self.clear_generation_t(t)

        # create a single sample result, with start time correction
        sample = self.create_sample(id_results, n)

        return sample

    def start_generation_t(
            self, n: int, t: int, simulate_one: Callable, all_accepted: bool,
            is_prel: bool) -> None:
        """Start generation `t`."""
        ana_id = self.analysis_id

        # write initial values to pipeline
        pipeline = self.redis.pipeline()
        # initialize variables for time t
        self.redis.set(idfy(SSA, ana_id, t),
                       cloudpickle.dumps((simulate_one, self.sample_factory)))
        pipeline.set(idfy(N_EVAL, ana_id, t), 0)
        pipeline.set(idfy(N_ACC, ana_id, t), 0)
        pipeline.set(idfy(N_REQ, ana_id, t), n)
        pipeline.set(idfy(N_FAIL, ana_id, t), 0)
        # encode as int
        pipeline.set(idfy(ALL_ACCEPTED, ana_id, t), int(all_accepted))
        pipeline.set(idfy(N_WORKER, ana_id, t), 0)
        pipeline.set(idfy(BATCH_SIZE, ana_id, t), self.batch_size)
        pipeline.set(idfy(IS_PREL, ana_id, t), int(is_prel))  # encode as int

        # update the current-generation variable
        pipeline.set(idfy(GENERATION, ana_id), t)

        # execute all commands
        pipeline.execute()

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
        pipeline = self.redis.pipeline()
        pipeline.delete(idfy(SSA, ana_id, t))
        pipeline.delete(idfy(N_EVAL, ana_id, t))
        pipeline.delete(idfy(N_ACC, ana_id, t))
        pipeline.delete(idfy(N_REQ, ana_id, t))
        pipeline.delete(idfy(N_FAIL, ana_id, t))
        pipeline.delete(idfy(ALL_ACCEPTED, ana_id, t))
        pipeline.delete(idfy(N_WORKER, ana_id, t))
        pipeline.delete(idfy(BATCH_SIZE, ana_id, t))
        pipeline.delete(idfy(QUEUE, ana_id, t))
        pipeline.execute()

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

    def maybe_start_next_generation(
            self, t: int, n: int, id_results: List,
            ana_vars: AnalysisVars) -> None:
        """Start the next generation already, if that looks reasonable.

        Parameters
        ----------
        t: The current time.
        n: The current population size.
        id_results: The so-far returned samples.
        ana_vars: Analysis variables.

        Note
        ----
        Currently we assume that
        * `n` is fixed,
        * distance and epsilon scheme are non-adaptive.
        """
        # not in a look-ahead mood
        if not self.look_ahead:
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
        acceptance_rate = len(population.get_list()) / nr_evaluations

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
            t=t+1, population=population, ana_vars=ana_vars)

        # head-start the next generation
        #  all_accepted is most certainly False for t>0
        self.start_generation_t(
            n=n, t=t+1, simulate_one=simulate_one_prel,
            all_accepted=False, is_prel=True)

    def stop(self):
        """Stop potentially still ongoing sampling."""
        # delete ids specifying the current analysis
        self.redis.delete(ANALYSIS_ID)
        self.redis.delete(idfy(GENERATION, self.analysis_id))
        # note: the other ana_id-t-specific variables are not deleted, as these
        #  do not hurt anyway and could potentially make the workers fail


def create_preliminary_simulate_one(
        t, population, ana_vars: AnalysisVars) -> Callable:
    """Create a preliminary simulate_one function for generation `t+1`.

    Based on preliminary results, update transitions, distance function,
    epsilon threshold etc., and return a function that samples parameters,
    simulates data and checks their preliminary acceptance.
    As the actual acceptance criteria may be different, samples generated by
    this function must be checked anew afterwards.

    Parameters
    ----------
    t: The time index for which to create the function (i.e. call with t+1).
    population: The preliminary population.
    ana_vars: The analysis variables.

    Returns
    -------
    simulate_one: The preliminary sampling function.
    """
    model_probabilities = population.get_model_probabilities()

    # create deep copies of potentially modified objects
    transitions = copy.deepcopy(ana_vars.transitions)
    distance_function = copy.deepcopy(ana_vars.distance_function)
    eps = copy.deepcopy(ana_vars.eps)
    acceptor = copy.deepcopy(ana_vars.acceptor)

    # fit transitions
    for m in population.get_alive_models():
        parameters, w = population.get_distribution(m)
        transitions[m].fit(parameters, w)

    # TODO fit distance, eps, acceptor

    return create_simulate_function(
        t=t, model_probabilities=model_probabilities,
        model_perturbation_kernel=ana_vars.model_perturbation_kernel,
        transitions=transitions, model_prior=ana_vars.model_prior,
        parameter_priors=ana_vars.parameter_priors,
        nr_samples_per_parameter=ana_vars.nr_samples_per_parameter,
        models=ana_vars.models, summary_statistics=ana_vars.summary_statistics,
        x_0=ana_vars.x_0, distance_function=distance_function,
        eps=eps, acceptor=acceptor,
    )
