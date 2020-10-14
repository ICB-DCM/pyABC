import numpy as np
import pickle
from time import sleep
import cloudpickle
import copy
from redis import StrictRedis
from typing import Callable, List, Tuple
from jabbar import jabbar

from ...sampler import Sampler, Sample
from .cmd import (SSA, N_EVAL, N_ACC, N_REQ, ALL_ACCEPTED,
                  N_WORKER, QUEUE, MSG, START,
                  SLEEP_TIME, BATCH_SIZE, IS_PREL, GENERATION, idfy)
from .redis_logging import logger


class RedisEvalParallelSampler(Sampler):
    """
    Redis based low latency sampler.
    This sampler is well performing in distributed environments.
    It is usually faster than the
    :class:`pyabc.sampler.DaskDistributedSampler` for
    short model evaluation runtimes. The longer the model evaluation times,
    the less the advantage becomes. It requires a running Redis server as
    broker.

    This sampler requires workers to be started via the command
    ``abc-redis-worker``.
    An example call might look like
    ``abc-redis-worker --host=123.456.789.123 --runtime=2h``
    to connect to a Redis server on IP ``123.456.789.123`` and to terminate
    the worker after finishing the first population which ends after 2 hours
    since worker start. So the actual runtime might be longer than 2h.
    See ``abc-redis-worker --help`` for its options.

    Use the command ``abc-redis-manager`` to retrieve info and stop the running
    workers.

    Start as many workers as you wish. Workers can be dynamically added
    during the ABC run.

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

    def sample_until_n_accepted(
            self, n, simulate_one, t, max_eval=np.inf, all_accepted=False,
            **kwargs):
        if self.generation_t_was_started(t):
            # update the SSA function
            self.redis.set(
                idfy(SSA, t),
                cloudpickle.dumps((simulate_one, self.sample_factory)))
            self.redis.set(idfy(IS_PREL, t), int(False))
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
                dump = self.redis.blpop(idfy(QUEUE, t))[1]
                # extract pickled object
                sample_with_id = pickle.loads(dump)
                # TODO check whether the acceptance criterion changed

                any_particle_accepted = False
                for result in sample_with_id[1]._particles:
                    if result.is_prel:

                        accepted_sum_stats = []
                        accepted_distances = []
                        accepted_weights = []
                        rejected_sum_stats = []
                        rejected_distances = []

                        for i_sum_stat, sum_stat in enumerate(result.accepted_sum_stats):
                            acc_res = kwargs['acceptor'](
                                distance_function=kwargs['distance_function'],
                                eps=kwargs['eps'],
                                x=sum_stat,
                                x_0=kwargs['x_0'],
                                t=t,
                                par=result.parameter)

                            if acc_res.accepted:
                                accepted_distances.append(acc_res.distance)
                                accepted_sum_stats.append(sum_stat)
                                accepted_weights.append(acc_res.weight)
                            else:
                                rejected_distances.append(acc_res.distance)
                                rejected_sum_stats.append(sum_stat)

                        result.accepted = len(accepted_distances) > 0

                        result.accepted_distances = accepted_distances
                        result.accepted_sum_stats = accepted_sum_stats
                        result.accepted_weights = accepted_weights
                        result.rejected_distances = rejected_distances
                        result.rejected_sum_stats = rejected_sum_stats

                        if result.accepted:
                            self.redis.incr(idfy(N_ACC, t), 1)
                            any_particle_accepted = True

                            # TODO Update rejected sumstats & distances

                if any_particle_accepted:
                    # append to collected results
                    id_results.append(sample_with_id)
                    bar.inc()

        # maybe head-start the next generation already
        self.maybe_start_next_generation(
            t=t, n=n, id_results=id_results, **kwargs)

        # wait until all workers done
        while int(self.redis.get(idfy(N_WORKER, t)).decode()) > 0:
            sleep(SLEEP_TIME)

        # make sure all results are collected
        while self.redis.llen(idfy(QUEUE, t)) > 0:
            id_results.append(
                pickle.loads(self.redis.blpop(idfy(QUEUE, t))[1]))

        # set total number of evaluations
        self.nr_evaluations_ = int(self.redis.get(idfy(N_EVAL, t)).decode())

        # remove all time-specific variables
        self.clear_generation_t(t)

        # create a single sample result, with start time correction
        sample = self.create_sample(id_results, n)

        return sample

    def start_generation_t(
            self, n: int, t: int, simulate_one: Callable, all_accepted: bool,
            is_prel: bool) -> None:
        """Start generation `t`."""
        # write initial values to pipeline
        pipeline = self.redis.pipeline()

        # initialize variables for time t
        self.redis.set(idfy(SSA, t),
                       cloudpickle.dumps((simulate_one, self.sample_factory)))
        pipeline.set(idfy(N_EVAL, t), 0)
        pipeline.set(idfy(N_ACC, t), 0)
        pipeline.set(idfy(N_REQ, t), n)
        pipeline.set(idfy(ALL_ACCEPTED, t), int(all_accepted))  # encode as int
        pipeline.set(idfy(N_WORKER, t), 0)
        pipeline.set(idfy(BATCH_SIZE, t), self.batch_size)
        pipeline.set(idfy(IS_PREL, t), int(is_prel))  # encode as int

        # update the current generation variable
        pipeline.set(GENERATION, t)

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
        return self.redis.exists(idfy(N_REQ, t))

    def clear_generation_t(self, t: int) -> None:
        """Clean up after generation `t` has finished.

        Parameters
        ----------
        t: The time for which to clear.
        """
        # delete keys from pipeline
        pipeline = self.redis.pipeline()
        pipeline.delete(idfy(SSA, t))
        pipeline.delete(idfy(N_EVAL, t))
        pipeline.delete(idfy(N_ACC, t))
        pipeline.delete(idfy(N_REQ, t))
        pipeline.delete(idfy(ALL_ACCEPTED, t))
        pipeline.delete(idfy(N_WORKER, t))
        pipeline.delete(idfy(BATCH_SIZE, t))
        pipeline.delete(idfy(QUEUE, t))
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
            self, t, n, id_results,
            eps, min_eps, stop_if_single_model_alive, min_acceptance_rate,
            max_t, **kwargs) -> None:
        """Start the next generation already, if that looks reasonable.

        Parameters
        ----------
        t: The current time.
        n: The current population size.
        id_results: The so-far returned samples.
        eps: The epsilon threshold scheme.
        min_eps: The minimum epsilon value.
        stop_if_single_model_alive: Whether to stop with only one model left.
        min_acceptance_rate: The minimum acceptance rate.
        max_t: The maximum generation time index.

        Note
        ----
        Currently we assume that
        * `n` is fixed,
        * distance and epsilon scheme are non-adaptive.
        """
        # not in a look-ahead mood
        if not self.look_ahead:
            return

        from pyabc.util import termination_criteria_fulfilled

        # create a result sample
        sample = self.create_sample(id_results, n)
        # copy as we modify the particles
        sample = copy.deepcopy(sample)

        # extract population
        population = sample.get_accepted_population()

        # acceptance rate
        nr_evaluations = int(self.redis.get(idfy(N_EVAL, t)).decode())
        acceptance_rate = len(population.get_list()) / nr_evaluations

        # check if any termination criterion (based on the current data)
        #  is likely to be fulfilled after the current generation
        if termination_criteria_fulfilled(
                current_eps=eps(t), min_eps=min_eps,
                stop_if_single_model_alive=stop_if_single_model_alive,
                nr_of_models_alive=population.nr_of_models_alive(),
                acceptance_rate=acceptance_rate,
                min_acceptance_rate=min_acceptance_rate, t=t, max_t=max_t):
            return

        # create a preliminary simulate_one function
        simulate_one_prel = _create_preliminary_simulate_one(
            t=t+1, sample=sample,
            eps=eps, **kwargs)

        # head-start the next generation
        #  all_accepted is most certainly False for t>0
        self.start_generation_t(
            n=n, t=t+1, simulate_one=simulate_one_prel,
            all_accepted=False, is_prel=True)


def _create_preliminary_simulate_one(
        t, sample,
        model_perturbation_kernel, transitions, model_prior, parameter_priors,
        nr_samples_per_parameter, models, summary_statistics, x_0,
        distance_function, eps, acceptor) -> Callable:
    """Create a preliminary simulate_one function for generation `t+1`.

    Based on preliminary results, update transitions, distance function,
    epsilon threshold etc., and return a function that samples parameters,
    simulates data and checks their preliminary acceptance.
    As the actual acceptance criteria may be different, samples generated by
    this function must be checked anew afterwards.

    Parameters
    ----------
    t: The time index for which to create the function (i.e. call with t+1).
    sample: The preliminary sample object.
    model_perturbation_kernel: The used model perturbation kernel.
    transitions:
        The parameter transition kernels. A deep copy of them is created here,
        as we need to fit them to the preliminary data.
    model_prior: The model prior.
    parameter_priors: The parameter priors.
    nr_samples_per_parameter: Number of samples per parameter.
    models: The models.
    summary_statistics: The summary statistics function.
    x_0: The observed summary statistics.
    distance_function:
        The distance function. A deep copy of it is created here, as we need
        to fit it to the preliminary data.
    eps:
        The epsilon threshold. A deep copy of it is created here, as we need
        to fit it to the preliminary data.
    acceptor:
        The acceptor. A deep copy of it is created here, as we need
        to fit it to the preliminary data.

    Returns
    -------
    simulate_one: The preliminary sampling function.
    """
    # check whether to maybe stop
    from pyabc.util import create_prel_simulate_function

    # extract accepted population
    population = sample.get_accepted_population()

    model_probabilities = population.get_model_probabilities()

    # create deep copies of potentially modified objects
    transitions = copy.deepcopy(transitions)
    distance_function = copy.deepcopy(distance_function)
    eps = copy.deepcopy(eps)
    acceptor = copy.deepcopy(acceptor)

    # fit transitions
    for m in population.get_alive_models():
        parameters, w = population.get_distribution(m)
        transitions[m].fit(parameters, w)

    # TODO fit distance, eps, acceptor

    return create_prel_simulate_function(
        t=t, model_probabilities=model_probabilities,
        model_perturbation_kernel=model_perturbation_kernel,
        transitions=transitions, model_prior=model_prior,
        parameter_priors=parameter_priors,
        nr_samples_per_parameter=nr_samples_per_parameter,
        models=models, summary_statistics=summary_statistics,
        x_0=x_0, distance_function=distance_function,
        eps=eps, acceptor=acceptor,
    )
