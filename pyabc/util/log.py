"""Logging utility functions."""

import numpy as np


def log_samples(
    t: int,
    sumstats: np.ndarray,
    parameters: np.ndarray,
    weights: np.ndarray,
    log_file: str,
):
    """Save samples to file, in npy format.

    Files will be created of name "{log_file}_{t}_{var}.npy",
    with var in sumstats, parameters, weights.

    Parameters
    ----------
    t: Time to save for.
    sumstats: Summary statistics, shape (n_sample, n_in).
    parameters: Parameters, shape (n_sample, n_par).
    weights: Importance sampling weights, shape (n_sample,).
    log_file: Log file base name. If None, no logs are created.
    """
    if log_file is None:
        return

    # uniquely define log file
    log_file += f"_{t}"

    for key, var in [
        ("sumstats", sumstats),
        ("parameters", parameters),
        ("weights", weights),
    ]:
        np.save(log_file + f"_{key}", var, allow_pickle=False)
