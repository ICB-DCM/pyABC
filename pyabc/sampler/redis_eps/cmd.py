"""Constants."""

# mostly communication keys

# id of the current analysis
ANALYSIS_ID = "analysis_id"
# generation index
GENERATION = "generation"
# dynamic or static
MODE = "mode"
STATIC = "static"
DYNAMIC = "dynamic"
# the queue to return results through
QUEUE = "queue"
# collection of active indices
ACTIVE_SET = "active_set"

# lock for evaluation counters
EVAL_LOCK = "eval_lock"
# lock for accessing the active set data structure
ACTIVE_SET_LOCK = "active_set_lock"

# default sleep time
SLEEP_TIME = .1

# message channel
MSG = "msg_pubsub"
# start and stop messages
START = "start"
STOP = "stop"

# whether all particles will be accepted
ALL_ACCEPTED = "all_accepted"
# wrap of the main inference routine
SSA = "sample_simulate_accept"
# whether look-ahead mode is to be employed
IS_LOOK_AHEAD = "is_look_ahead"
# maximum number of evaluations in look-ahead mode
MAX_N_EVAL_LOOK_AHEAD = "max_n_eval_look_ahead"
# batch size to use
BATCH_SIZE = "batch_size"

# counters
#  evaluations
N_EVAL = "n_eval"
#  acceptances
N_ACC = "n_acc"
#  required particles (population size)
N_REQ = "n_req"
#  failures
N_FAIL = "n_fail"
#  active workers
N_WORKER = "n_worker"
#  lookahead evaluations
N_LOOKAHEAD_EVAL = "n_lookahead_eval"
#  jobs (static only)
N_JOB = "n_job"


def idfy(var: str, *args):
    """Append ids to the variable."""
    for arg in args:
        var += '_' + str(arg)
    return var
