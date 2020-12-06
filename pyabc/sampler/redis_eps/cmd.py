"""Communication keys"""
QUEUE = "queue"
N_EVAL = "n_eval"
N_ACC = "n_acc"
N_REQ = "n_req"
N_FAIL = "n_fail"
ALL_ACCEPTED = "all_accepted"
SSA = "sample_simulate_accept"
N_WORKER = "n_workers"
IS_PREL = "is_prel"
ANALYSIS_ID = "analysis_id"
GENERATION = "generation"

MSG = "msg_pubsub"
START = "start"
STOP = "stop"
BATCH_SIZE = "batch_size"
SLEEP_TIME = .1


def idfy(var: str, *args):
    """Append ids to the variable."""
    for arg in args:
        var += '_' + str(arg)
    return var
