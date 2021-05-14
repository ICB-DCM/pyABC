"""Function to work on a population in static mode."""

import sys
from redis import StrictRedis
import pickle
import cloudpickle
from time import time
import logging

from .cmd import (N_EVAL, N_ACC, N_FAIL, N_JOB, N_WORKER, SSA, QUEUE,
                  ANALYSIS_ID, idfy)
from .cli import KillHandler

logger = logging.getLogger("ABC.Sampler")


def announce_work(work_on_population):
    """Handle the worker counter."""
    def wrapper(
            analysis_id: str, t: int, redis: StrictRedis,
            kill_handler: KillHandler, **kwargs):
        # notify sign up as worker
        n_worker = redis.incr(idfy(N_WORKER, analysis_id, t))
        logger.info(
            f"Begin generation {t}. I am worker {n_worker}")
        # don't be killed during work
        kill_handler.exit = False

        # do the actual work
        ret = work_on_population(
            analysis_id=analysis_id, t=t, redis=redis,
            kill_handler=kill_handler, n_worker=n_worker, **kwargs)

        # notify end work
        redis.decr(idfy(N_WORKER, analysis_id, t))
        # allow to kill again
        kill_handler.exit = True

        # return whatever the method wants to return
        return ret
    return wrapper


@announce_work
def work_on_population_static(
        analysis_id: str,
        t: int,
        redis: StrictRedis,
        catch: bool,
        start_time: float,
        max_runtime_s: float,
        kill_handler: KillHandler,
        n_worker: int):
    """Work on population in static mode.
    Here the actual sampling happens.
    """
    def get_int(var: str):
        """Convenience function to read an int variable."""
        return int(redis.get(idfy(var, ana_id, t)).decode())

    # set timers
    population_start_time = time()
    cumulative_simulation_time = 0

    # short-form
    ana_id = analysis_id

    # extract bytes
    ssa_b = redis.get(idfy(SSA, ana_id, t))

    if ssa_b is None:
        # no more work needed in the meantime
        return

    # convert from bytes
    simulate_one, sample_factory = pickle.loads(ssa_b)

    # count simulations
    internal_counter = 0

    while True:
        with redis.lock('worker'):
            # check whether there is work to be done
            n_job_b = redis.get(idfy(N_JOB, ana_id, t))
            if n_job_b is None or int(n_job_b.decode()) <= 0:
                population_total_time = time() - population_start_time
                logger.info(
                    "I'm a sad jobless worker. "
                    f"Finished generation {t}, did {internal_counter} "
                    "samples. "
                    f"Simulation time: {cumulative_simulation_time:.2f}s, "
                    f"total time {population_total_time:.2f}.")
                return

            # decrease job counter
            redis.decr(idfy(N_JOB, ana_id, t))

        # sample until one simulation gets accepted
        sample = sample_factory()

        while True:
            # check whether the process was externally asked to stop
            if kill_handler.killed:
                logger.info(
                    f"Worker {n_worker} received stop signal. "
                    "Terminating in the middle of a population "
                    f"after {internal_counter} samples.")
                # notify quit (manually here as we call exit)
                redis.decr(idfy(N_WORKER, ana_id, t))
                redis.incr(idfy(N_JOB, ana_id, t))
                sys.exit(0)

            # check whether time's up
            current_runtime = time() - start_time
            if current_runtime > max_runtime_s:
                logger.info(
                    f"Worker {n_worker} stops during population because "
                    f"runtime {current_runtime} exceeds "
                    f"max runtime {max_runtime_s}")
                # return to task queue
                redis.incr(idfy(N_JOB, ana_id, t))
                return

            # check whether the analysis was terminated or replaced by a new
            #  one
            ana_id_new_b = redis.get(ANALYSIS_ID)
            if ana_id_new_b is None or str(ana_id_new_b.decode()) != ana_id:
                logger.info(
                    f"Worker {n_worker} stops during population because "
                    "the analysis seems to have been stopped.")
                # return to task queue
                redis.incr(idfy(N_JOB, ana_id, t))
                return

            # increase global evaluation counter
            redis.incr(idfy(N_EVAL, ana_id, t))
            # increase internal evaluation counter
            internal_counter += 1

            # timer for current simulation until batch_size acceptances
            this_sim_start = time()
            try:
                # simulate
                new_sim = simulate_one()
            except Exception as e:
                logger.warning(f"Redis worker number {n_worker} failed. "
                               f"Error message is: {e}")
                # increment the failure counter
                redis.incr(idfy(N_FAIL, ana_id, t), 1)
                if not catch:
                    raise e
                continue

            # update total simulation-specific time
            cumulative_simulation_time += time() - this_sim_start

            # append to current sample
            sample.append(new_sim)
            # check for acceptance
            if new_sim.accepted:
                # serialize simulation
                dump = cloudpickle.dumps(sample)
                # put on pipe
                (redis.pipeline()
                 .incr(idfy(N_ACC, ana_id, t))
                 .rpush(idfy(QUEUE, ana_id, t), dump)
                 .execute())

                # upon success, leave the loop and check the job queue again
                break
