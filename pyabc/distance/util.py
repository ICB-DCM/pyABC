import numpy as np
from logging import Logger
from typing import Collection, Dict, List, Union
import collections.abc
import os

from ..storage import load_dict_from_json, save_dict_to_json


def bound_weights(w: np.ndarray, max_weight_ratio: float) -> np.ndarray:
    """
    Bound all weights to `max_weight_ratio` times the minimum
    non-zero absolute weight, if `max_weight_ratio` is not None.

    While this is usually not required in practice, it is theoretically
    necessary that the ellipses are not arbitrarily eccentric, in order
    to ensure convergence.
    """
    if max_weight_ratio is None:
        return w

    # find minimum absolute weight > 0
    min_w = np.min(np.abs(w[~np.isclose(w, 0)]))

    # cap weights
    w[w / min_w > max_weight_ratio] = min_w * max_weight_ratio

    return w


def log_weights(
    t: int,
    weights: Dict[int, np.ndarray],
    keys: List[str],
    label: str,
    log_file: str,
    logger: Logger,
) -> None:
    """Log weights.

    Parameters
    ----------
    t: Time point to log for.
    weights: All weights.
    keys: Summary statistic keys.
    label: Label to identify different weight types.
    log_file: File to log formatted output to.
    logger: Logger for debugging purposes.
    """
    # create weights dictionary with labels
    weights = {key: val for key, val in zip(keys, weights[t])}

    vals = [f"'{key}': {val:.4e}" for key, val in weights.items()]
    logger.debug(f"{label} weights[{t}] = {{{', '.join(vals)}}}")

    if log_file:
        # read in file
        dct = {}
        if os.path.exists(log_file):
            dct = load_dict_from_json(file_=log_file)
        # add column
        if t in dct:
            logger.warning(
                f"Time {t} already in log file {log_file}. "
                "Overwriting, but this looks suspicious.",
            )
        dct[t] = weights
        # save to file
        save_dict_to_json(dct, log_file)


def to_fit_ixs(ixs: Union[Collection, int]) -> set:
    """Input to collection of time indices when to fit."""
    # convert inf or int to range
    if not isinstance(ixs, collections.abc.Collection):
        if ixs == np.inf:
            ixs = {0, np.inf}
        else:
            # create set {0, ..., ixs-1}, # = ixs
            ixs = set(range(0, int(ixs)))
    return set(ixs)
