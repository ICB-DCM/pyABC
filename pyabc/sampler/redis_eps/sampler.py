import numpy as np
import pickle
import copy
import cloudpickle
from redis import StrictRedis
from ...sampler import Sampler
from .cmd import (SSA, N_EVAL, N_ACC, N_REQ, ALL_ACCEPTED,
                  N_WORKER, QUEUE, MSG, START,
                  BATCH_SIZE, IS_PREL, GENERATION)
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

    host: str, optional
        IP address or name of the Redis server.
        Default is "localhost".

    port: int, optional
        Port of the Redis server.
        Default is 6379.

    password: str, optional
        Password for a protected server. Default is None (no protection).

    batch_size: int, optional
        Number of model evaluations the workers perform before contacting
        the REDIS server. Defaults to 1. Increase this value if model
        evaluation times are short or the number of workers is large
        to reduce communication overhead.
    """

    def __init__(self,
                 host: str = "localhost",
                 port: int = 6379,
                 password: str = None,
                 batch_size: int = 1):
        super().__init__()
        logger.debug(
            f"Redis sampler: host={host} port={port}")
        # handles the connection to the redis-server
        self.redis = StrictRedis(host=host, port=port, password=password)
        self.batch_size = batch_size

    def n_worker(self):
        """
        Get the number of connected workers.

        Returns
        -------

        Number of workers connected.
        """
        return self.redis.pubsub_numsub(MSG)[0][-1]

    """
    def sample_until_n_accepted(
            self, n, simulate_one, max_eval=np.inf, all_accepted=False):
        # open pipeline
        pipeline = self.redis.pipeline()

        # write initial values to pipeline
        self.redis.set(
            SSA, cloudpickle.dumps((simulate_one, self.sample_factory)))
        pipeline.set(N_EVAL, 0)
        pipeline.set(N_ACC, 0)
        pipeline.set(N_REQ, n)
        pipeline.set(ALL_ACCEPTED, int(all_accepted))  # encode as int
        pipeline.set(N_WORKER, 0)
        pipeline.set(BATCH_SIZE, self.batch_size)
        # delete previous results
        pipeline.delete(QUEUE)
        # execute all commands
        pipeline.execute()

        id_results = []

        # publish start message
        self.redis.publish(MSG, START)

        # wait until n acceptances
        while len(id_results) < n:
            # pop result from queue, block until one is available
            dump = self.redis.blpop(QUEUE)[1]
            # extract pickled object
            particle_with_id = pickle.loads(dump)
            # append to collected results
            id_results.append(particle_with_id)

        # wait until all workers done
        while int(self.redis.get(N_WORKER).decode()) > 0:
            sleep(SLEEP_TIME)

        # make sure all results are collected
        while self.redis.llen(QUEUE) > 0:
            id_results.append(pickle.loads(self.redis.blpop(QUEUE)[1]))

        # set total number of evaluations
        self.nr_evaluations_ = int(self.redis.get(N_EVAL).decode())

        # delete keys from pipeline
        pipeline = self.redis.pipeline()
        pipeline.delete(SSA)
        pipeline.delete(N_EVAL)
        pipeline.delete(N_ACC)
        pipeline.delete(N_REQ)
        pipeline.delete(ALL_ACCEPTED)
        pipeline.delete(BATCH_SIZE)
        pipeline.execute()

        # avoid bias toward short running evaluations (for
        # dynamic scheduling)
        id_results.sort(key=lambda x: x[0])
        id_results = id_results[:n]

        results = [res[1] for res in id_results]

        # create 1 to-be-returned sample from results
        sample = self._create_empty_sample()
        for j in range(n):
            sample += results[j]

        return sample
    """

    def start_pipeline_for_gen_t(self, n, t, simulate_one, all_accepted, is_prel):
        """

        :param n: number of particles
        :param t: generation indicator
        :param simulate_one:
        :param all_accepted:
        :param is_prel: boolean
            indicates whether the simulate_one function is the final one
        :return:

        """
        # write initial values to pipeline
        pipeline = self.redis.pipeline()
        self.redis.set(SSA + "_" + str(t), cloudpickle.dumps((simulate_one, self.sample_factory)))
        pipeline.set(N_EVAL + "_" + str(t), 0)
        pipeline.set(N_ACC + "_" + str(t), 0)
        pipeline.set(N_REQ + "_" + str(t), n)
        pipeline.set(ALL_ACCEPTED + "_" + str(t), int(all_accepted))  # encode as int
        pipeline.set(N_WORKER + "_" + str(t), 0)
        pipeline.set(BATCH_SIZE, self.batch_size)
        pipeline.set(IS_PREL + "_" + str(t), int(is_prel))
        pipeline.set(GENERATION, t)

        # execute all commands
        pipeline.execute()

        # publish start message

        self.redis.publish(MSG, START)
        print("pipeline start gen t="+str(t))

    def sample_parallel_until_n_accepted(self,
                                         n,
                                         t,
                                         simulate_one,
                                         abcsmc,
                                         max_eval=np.inf,
                                         all_accepted=False
                                         ):

        pipeline = self.redis.pipeline()

        # debug
        print("start t=" + str(t))

        if t == 0:

            # open pipeline
            self.start_pipeline_for_gen_t(n, t, simulate_one, all_accepted, False)

        else:
            self.redis.set(SSA, cloudpickle.dumps((simulate_one, self.sample_factory)))
            pipeline.set(IS_PREL + "_" + str(t), False)

            # is this enough or do i have to completely re-execute the pipeline?

        id_results = []

        # debug
        print("first n acc loop start t=" + str(t))

        # wait until n acceptances
        while len(id_results) < n:
            # pop result from queue, block until one is available

            # debug
            print(str(len(id_results))+"th result")

            dump = self.redis.blpop(QUEUE + "_" + str(t))[1]

            print("after blpop")

            if t == 0:
                # extract pickled object
                particle_with_id = pickle.loads(dump)
                # append to collected results
                id_results.append(particle_with_id)
            else:
                distance = dump.accepted_distances
                if distance <= abcsmc.eps:
                    # extract pickled object
                    particle_with_id = pickle.loads(dump)
                    # append to collected results
                    id_results.append(particle_with_id)
                else:
                    pipeline.decr(N_ACC + "_" + str(t), 1)
                    dump.accepted = False

        # debug
        print("first n acc loop end " + str(t))

        prel_results = [res[1] for res in id_results]
        prel_sample = self._create_empty_sample()
        for j in range(n):
            prel_sample += prel_results[j]

        self.nr_evaluations_ = int(self.redis.get(N_EVAL + "_" + str(t)).decode())

        # create new simulate_one
        prel_transitions = copy.deepcopy(abcsmc.transitions)
        for m in abcsmc.history.alive_models(t - 1):
            particles, w = abcsmc.history.get_distribution(m, t - 1)
            prel_transitions[m].fit(particles, w)
        prel_transitions_pdf = abcsmc._create_transition_pdf(t + 1, prel_transitions)

        prior_pdf = abcsmc.prior_pdf
        prel_weight_function = self._create_prel_weight_function(abcsmc, prior_pdf, prel_transitions_pdf)

        prel_simulate_one = self._create_prel_simulate_one(t,
                                                           abcsmc,
                                                           prel_transitions,
                                                           prel_weight_function)

        self.start_pipeline_for_gen_t(n, t + 1, prel_simulate_one, all_accepted, True)
        """
        id_results + "_" + str(t+1) = []

        # while old pipeline finishes, begin new
        while int(self.redis.get(N_WORKER + "_" + str(t)).decode()) > 0:
            # pop result from queue, block until one is available
            dump = self.redis.blpop(QUEUE + "_" + str(t + 1))[1]
            # extract pickled object
            particle_with_id + "_" + str(t + 1) = pickle.loads(dump)
            # append to collected results
            id_results + "_" + str(t + 1).append(particle_with_id + "_" + str(t + 1))
        """

        # debug
        print("collect rest " + str(t))

        # make sure all results are collected
        while self.redis.llen(QUEUE + "_" + str(t)) > 0:
            id_results.append(pickle.loads(self.redis.blpop(QUEUE + "_" + str(t))[1]))

        # set total number of evaluations
        self.nr_evaluations_ = int(self.redis.get(N_EVAL + "_" + str(t)).decode())

        # delete keys from pipeline
        pipeline.delete(SSA + "_" + str(t))
        pipeline.delete(N_EVAL + "_" + str(t))
        pipeline.delete(N_ACC + "_" + str(t))
        pipeline.delete(N_REQ + "_" + str(t))
        pipeline.delete(ALL_ACCEPTED + "_" + str(t))
        pipeline.delete(IS_PREL + "_" + str(t))
        pipeline.delete(BATCH_SIZE)
        pipeline.execute()

        # avoid bias toward short running evaluations (for
        # dynamic scheduling)
        id_results.sort(key=lambda x: x[0])
        id_results = id_results[:n]

        results = [res[1] for res in id_results + "_" + str(t)]

        # create 1 to-be-returned sample from results
        sample = self._create_empty_sample()
        for j in range(n):
            sample += results[j]

        # debug
        print("end " + str(t))

        return sample

    def _create_prel_weight_function(self, abcsmc, prior_pdf, transition_pdf):

        nr_samples_per_parameter = abcsmc.population_size.nr_samples_per_parameter

        def prel_weight_function(
                distance_list, m_ss, theta_ss, acceptance_weights):
            prior_pd = prior_pdf(m_ss, theta_ss)

            transition_pd = transition_pdf(m_ss, theta_ss)
            # account for stochastic acceptance
            acceptance_weight = np.prod(acceptance_weights)
            # account for multiple tries
            fraction_accepted_runs_for_single_parameter = \
                len(distance_list) / nr_samples_per_parameter

            # calculate weight
            weight = (prior_pd * acceptance_weight
                      * fraction_accepted_runs_for_single_parameter
                      / transition_pd)
            return weight

        return prel_weight_function

    def _create_prel_simulate_one(self, t,
                                  abcsmc,
                                  prel_transitions,
                                  prel_weight_function):
        model_probabilities = abcsmc.history.get_model_probabilities(t - 1)
        m = np.array(model_probabilities.index)
        p = np.array(model_probabilities.p)

        model_prior = abcsmc.model_prior
        parameter_priors = abcsmc.parameter_priors
        model_perturbation_kernel = abcsmc.model_perturbation_kernel
        nr_samples_per_parameter = abcsmc.population_size.nr_samples_per_parameter
        models = abcsmc.models
        summary_statistics = abcsmc.summary_statistics
        distance_function = abcsmc.distance_function
        eps = abcsmc.eps
        acceptor = abcsmc.acceptor
        x_0 = abcsmc.x_0

        def prel_simulate_one():
            parameter = abcsmc._generate_valid_proposal(
                t, m, p,
                model_prior,
                parameter_priors,
                model_perturbation_kernel,
                prel_transitions)
            particle = abcsmc._evaluate_proposal(
                *parameter,
                t,
                nr_samples_per_parameter,
                models,
                summary_statistics,
                distance_function,
                eps,
                acceptor,
                x_0,
                prel_weight_function)
            return particle

        return prel_simulate_one
