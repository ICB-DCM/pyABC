"""Function to work on a population in dynamic mode."""

import sys
from redis import StrictRedis
import cloudpickle as pickle
from time import sleep, time
import logging

from ..util import any_particle_preliminary
from .cmd import (
    N_EVAL, N_ACC, N_REQ, N_FAIL, ALL_ACCEPTED, N_WORKER, N_LOOKAHEAD_EVAL,
    SSA, QUEUE, BATCH_SIZE, IS_LOOK_AHEAD, ANALYSIS_ID, MAX_N_EVAL_LOOK_AHEAD,
    SLEEP_TIME, DONE_IXS, idfy)
from .cli import KillHandler

logger = logging.getLogger("ABC.Sampler")


def work_on_population_dynamic(
        analysis_id: str,
        t: int,
        redis: StrictRedis,
        catch: bool,
        start_time: float,
        max_runtime_s: float,
        kill_handler: KillHandler):
    """Work on population in dynamic mode.
    Here the actual sampling happens.
    """
    # short-form
    ana_id = analysis_id

    def get_int(var: str):
        """Convenience function to read an int variable."""
        return int(redis.get(idfy(var, ana_id, t)).decode())

    # set timers
    population_start_time = time()
    cumulative_simulation_time = 0

    # read from pipeline
    pipeline = redis.pipeline()

    # extract bytes
    (ssa_b, batch_size_b, all_accepted_b, is_look_ahead_b,
     max_eval_look_ahead_b) = (
        pipeline.get(idfy(SSA, ana_id, t))
        .get(idfy(BATCH_SIZE, ana_id, t))
        .get(idfy(ALL_ACCEPTED, ana_id, t))
        .get(idfy(IS_LOOK_AHEAD, ana_id, t))
        .get(idfy(MAX_N_EVAL_LOOK_AHEAD, ana_id, t)).execute())

    # if the ssa object does not exist, something went wrong, return
    if ssa_b is None:
        return
    # notify sign up as worker
    n_worker = redis.incr(idfy(N_WORKER, ana_id, t))
    logger.info(
        f"Begin generation {t}, I am worker {n_worker}")

    # only allow stopping the worker at particular points
    kill_handler.exit = False

    # convert from bytes
    simulate_one, sample_factory = pickle.loads(ssa_b)
    batch_size = int(batch_size_b.decode())
    all_accepted = bool(int(all_accepted_b.decode()))
    is_look_ahead = bool(int(is_look_ahead_b.decode()))
    max_n_eval_look_ahead = float(max_eval_look_ahead_b.decode())

    # counter for number of simulations
    internal_counter = 0

    # create empty sample
    sample = sample_factory(is_look_ahead=is_look_ahead)

    # loop until no more particles required
    # all numbers are re-loaded in each iteration as they can dynamically
    #  update
    while get_int(N_ACC) < get_int(N_REQ) and (
            not all_accepted or
            get_int(N_EVAL) - get_int(N_FAIL) < get_int(N_REQ)):
        # check whether the process was externally asked to stop
        if kill_handler.killed:
            logger.info(
                f"Worker {n_worker} received stop signal. "
                "Terminating in the middle of a population "
                f"after {internal_counter} samples.")
            # notify quit
            redis.decr(idfy(N_WORKER, ana_id, t))
            sys.exit(0)

        # check whether time's up
        current_runtime = time() - start_time
        if current_runtime > max_runtime_s:
            logger.info(
                f"Worker {n_worker} stops during population because "
                f"runtime {current_runtime} exceeds "
                f"max runtime {max_runtime_s}")
            # notify quit
            redis.decr(idfy(N_WORKER, ana_id, t))
            # return to task queue
            return

        # check whether the analysis was terminated or replaced by a new one
        ana_id_new_b = redis.get(ANALYSIS_ID)
        if ana_id_new_b is None or str(ana_id_new_b.decode()) != ana_id:
            logger.info(
                f"Worker {n_worker} stops during population because "
                "the analysis seems to have been stopped.")
            # notify quit
            redis.decr(idfy(N_WORKER, ana_id, t))
            # return to task queue
            return

        # check if the analysis left the look-ahead mode
        if is_look_ahead and not bool(int(
                redis.get(idfy(IS_LOOK_AHEAD, ana_id, t)).decode())):
            # reload SSA object
            ssa_b = redis.get(idfy(SSA, ana_id, t))
            simulate_one, sample_factory = pickle.loads(ssa_b)
            # cache
            is_look_ahead = False
            # create new empty sample for clean split
            sample = sample_factory(is_look_ahead=is_look_ahead)

        # check if in look-ahead mode and should sleep
        if is_look_ahead and get_int(N_EVAL) >= max_n_eval_look_ahead:
            # sleep ... seconds
            sleep(SLEEP_TIME)
            continue

        # increase global evaluation counter (before simulation!)
        particle_max_id: int = redis.incr(
            idfy(N_EVAL, ana_id, t), batch_size)

        if is_look_ahead:
            # increment look-ahead evaluation counter
            redis.incr(idfy(N_LOOKAHEAD_EVAL, ana_id, t), batch_size)

        # timer for current simulation until batch_size acceptances
        this_sim_start = time()
        # collect accepted particles
        accepted_samples = []
        # whether any particle in this iteration is preliminary
        any_prel = False

        # make batch_size attempts
        for n_batched in range(batch_size):
            # increase evaluation counter
            internal_counter += 1
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

            # append to current sample
            sample.append(new_sim)
            # check for acceptance
            if new_sim.accepted:
                # The order of the IDs is reversed, but this does not
                #  matter. Important is only that the IDs are specified
                #  before the simulation starts

                # append to accepted list
                accepted_samples.append(
                    pickle.dumps((particle_max_id - n_batched, sample)))
                any_prel = any_prel or any_particle_preliminary(sample)
                # initialize new sample
                sample = sample_factory(is_look_ahead=is_look_ahead)

        # update total simulation-specific time
        cumulative_simulation_time += time() - this_sim_start

        # new pipeline
        pipeline = redis.pipeline()
        # push to pipeline if at least one sample got accepted
        if len(accepted_samples) > 0:
            # update particles counter if nothing is preliminary,
            #  otherwise final acceptance is done by the sampler
            if not any_prel:
                pipeline.incr(idfy(N_ACC, ana_id, t), len(accepted_samples))
            # note: samples are appended 1-by-1
            pipeline.rpush(idfy(QUEUE, ana_id, t), *accepted_samples)
        # append to list of done simulations
        pipeline.rpush(
            idfy(DONE_IXS, ana_id, t),
            *range(particle_max_id - batch_size + 1, particle_max_id + 1),
        )
        # execute all commands
        pipeline.execute()

    # end of sampling loop

    # notify quit
    redis.decr(idfy(N_WORKER, ana_id, t))
    kill_handler.exit = True
    population_total_time = time() - population_start_time
    logger.info(
        f"Finished generation {t}, did {internal_counter} samples. "
        f"Simulation time: {cumulative_simulation_time:.2f}s, "
        f"total time {population_total_time:.2f}.")
