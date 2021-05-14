"""Redis based static scheduling sampler."""

import numpy as np
import pickle
import cloudpickle
from time import sleep
import logging
from typing import Callable, List
from jabbar import jabbar

from ...sampler import Sample
from .cmd import (SSA, N_EVAL, N_ACC, N_REQ, N_FAIL, N_JOB, N_WORKER,
                  SLEEP_TIME, MODE, STATIC, QUEUE, MSG, START, GENERATION,
                  idfy)
from .sampler import RedisSamplerBase

logger = logging.getLogger("ABC.Sampler")


class RedisStaticSampler(RedisSamplerBase):
    """Redis based static scheduling sampler."""

    def sample_until_n_accepted(
            self, n, simulate_one, t, *,
            max_eval=np.inf, all_accepted=False, ana_vars=None):
        # get the analysis id
        ana_id = self.analysis_id

        # tell workers to start
        self.start_generation_t(n=n, t=t, simulate_one=simulate_one)

        # collect samples
        samples = []
        with jabbar(total=n, enable=self.show_progress, keep=False) as bar:
            while len(samples) < n:
                dump = self.redis.blpop(idfy(QUEUE, ana_id, t))[1]
                sample = pickle.loads(dump)
                if sum(particle.accepted
                       for particle in sample.particles) != 1:
                    # this should never happen
                    raise AssertionError(
                        "Expected exactly one accepted particle in sample.")
                samples.append(sample)
                bar.inc()

        # wait for all workers to join
        #  this is necessary for clear intermediate states
        while int(self.redis.get(idfy(N_WORKER, ana_id, t)).decode()) > 0:
            sleep(SLEEP_TIME)

        # set total number of evaluations
        self.nr_evaluations_ = int(
            self.redis.get(idfy(N_EVAL, ana_id, t)).decode())

        # remove all time-specific variables
        self.clear_generation_t(t)

        # create a single sample result, with start time correction
        sample = self.create_sample(samples, n)

        return sample

    def start_generation_t(
            self, n: int, t: int, simulate_one: Callable) -> None:
        """Start generation `t`."""
        ana_id = self.analysis_id

        # write initial values to pipeline
        (self.redis.pipeline()
         # initialize variables for time t
         .set(idfy(SSA, ana_id, t),
              cloudpickle.dumps((simulate_one, self.sample_factory)))
         .set(idfy(N_EVAL, ana_id, t), 0)
         # N_ACC here only serves for in-time debugging
         .set(idfy(N_ACC, ana_id, t), 0)
         .set(idfy(N_REQ, ana_id, t), n)
         .set(idfy(N_FAIL, ana_id, t), 0)
         .set(idfy(N_WORKER, ana_id, t), 0)
         .set(idfy(N_JOB, ana_id, t), n)
         .set(idfy(MODE, ana_id, t), STATIC)
         # update the current-generation variable
         .set(idfy(GENERATION, ana_id), t)
         # execute all commands
         .execute())

        # publish start message
        self.redis.publish(MSG, START)

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
         .delete(idfy(N_WORKER, ana_id, t))
         .delete(idfy(N_JOB, ana_id, t))
         .delete(idfy(MODE, ana_id, t))
         .delete(idfy(QUEUE, ana_id, t))
         .execute())

    def create_sample(self, samples: List[Sample], n: int) -> Sample:
        """Create a single sample result.
        Order the results by starting point to avoid a bias towards
        short-running simulations (dynamic scheduling).
        """
        if len(samples) != n:
            raise AssertionError(
                f"Expected {n} samples, got {len(samples)}.")

        # create 1 to-be-returned sample from results
        sample = self._create_empty_sample()
        for single_sample in samples:
            sample += single_sample

        return sample
