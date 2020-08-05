"""Communication keys"""
from typing import Any

QUEUE = "queue"
N_EVAL = "n_eval"
N_ACC = "n_acc"
N_REQ = "n_req"
ALL_ACCEPTED = "all_accepted"
SSA = "sample_simulate_accept"
N_WORKER = "n_workers"
IS_PREL = "is_prel"
GENERATION = "generation"

MSG = "msg_pubsub"
START = "start"
STOP = "stop"
BATCH_SIZE = "batch_size"
SLEEP_TIME = .1


def idfy(var: str, id_: Any):
    """Append the id to the variable."""
    return var + '_' + str(id_)
