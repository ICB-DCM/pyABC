import logging
import os
from typing import Callable, Dict, List, Sequence

import numpy as np

from ..storage import load_dict_from_json, save_dict_to_json

logger = logging.getLogger("ABC.Distance")


def bound_weights(
    w: np.ndarray,
    max_weight_ratio: float,
) -> np.ndarray:
    """
    Bound all weights to `max_weight_ratio` times the minimum
    non-zero absolute weight, if `max_weight_ratio` is not None.

    While this is usually not required in practice, it is theoretically
    necessary that the ellipses are not arbitrarily eccentric, in order
    to ensure convergence.

    Parameters
    ----------
    w: Weights.
    max_weight_ratio: Maximum ratio of maximum and minimum weight.

    Returns
    -------
    w: The bounded weights.
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
) -> None:
    """Log weights.

    Parameters
    ----------
    t: Time point to log for.
    weights: All weights.
    keys: Summary statistic keys.
    label: Label to identify different weight types.
    log_file: File to log formatted output to.
    """
    # create weights dictionary with labels
    weights = {key: val for key, val in zip(keys, weights[t])}

    vals = [f"'{key}': {val:.4e}" for key, val in weights.items()]
    logger.debug(f"{label} weights[{t}] = {{{', '.join(vals)}}}")

    if log_file:
        # read in file
        dct = {}
        if os.path.exists(log_file) and os.path.getsize(log_file) > 0:
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


def fd_nabla1_multi_delta(
    x: np.ndarray,
    fun: Callable,
    test_deltas: Sequence[float] = None,
) -> np.ndarray:
    """Calculate FD approximation to 1st order derivative (Jacobian/gradient)
    with automatic step size selection.

    Parameters
    ----------
    x: Parameter vector, shape (n_par,).
    fun: Function returning function values. Scalar- or vector-valued.
    test_deltas:
        Deltas to try. The optimal delta is chosen per coordinate.
        Defaults to [1e-1, 1e-2, ... 1e-8].

    Returns
    -------
    nabla_1:
        The FD approximation to the 1st order derivatives.
        Shape (n_par, ...) with ndim > 1 if `f_fval` is not scalar-valued.
    """
    if test_deltas is None:
        test_deltas = [10 ** (-i) for i in range(1, 9)]

    if len(test_deltas) == 1:
        test_deltas = test_deltas[0]

    if isinstance(test_deltas, float):
        delta = test_deltas * np.ones_like(x)
        return fd_nabla_1(x=x, fun=fun, delta_vec=delta)

    # calculate gradients for all deltas for all parameters
    nablas = []
    # iterate over deltas
    for delta in test_deltas:
        # calculate Jacobian with step size delta
        delta_vec = delta * np.ones_like(x)
        nabla = fd_nabla_1(x=x, fun=fun, delta_vec=delta_vec)

        nablas.append(nabla)

    # shape (n_delta, n_par, ...)
    nablas = np.array(nablas)

    # The stability vector is the the absolute difference of Jacobian
    #  entries towards smaller and larger deltas, thus indicating the
    #  change in the approximation when changing delta.
    # This is done separately for each parameter. Then, for each the delta
    #  with the minimal entry and thus the most stable behavior
    #  is selected.
    stab_vec = np.full(shape=nablas.shape, fill_value=np.nan)
    stab_vec[1:-1] = np.mean(
        np.abs(
            [nablas[2:] - nablas[1:-1], nablas[1:-1] - nablas[:-2]],
        ),
        axis=0,
    )
    # on the edge, just take the single neighbor
    stab_vec[0] = np.abs(nablas[1] - nablas[0])
    stab_vec[-1] = np.abs(nablas[-1] - nablas[-2])

    # if the function is tensor-valued, consider the maximum over all
    #  entries, to constrain the worst deviation
    if stab_vec.ndim > 2:
        # flatten all dimensions > 1
        stab_vec = stab_vec.reshape(
            stab_vec.shape[0], stab_vec.shape[1], -1
        ).max(axis=2)

    # minimum delta index for each parameter
    min_ixs = np.argmin(stab_vec, axis=0)

    # extract optimal deltas per parameter
    delta_opt = np.array([test_deltas[ix] for ix in min_ixs])

    # log
    logger.debug(f"Optimal FD delta: {delta_opt}")

    return fd_nabla_1(x=x, fun=fun, delta_vec=delta_opt)


def fd_nabla_1(
    x: np.ndarray,
    fun: Callable,
    delta_vec: np.ndarray,
) -> np.ndarray:
    """Calculate FD approximation to 1st order derivative (Jacobian/gradient).

    Parameters
    ----------
    x: Parameter vector, shape (n_par,).
    fun: Function returning function values. Scalar- or vector-valued.
    delta_vec: Step size vector, shape (n_par,).

    Returns
    -------
    nabla_1:
        The FD approximation to the 1st order derivatives.
        Shape (n_par, ...) with ndim > 1 if `f_fval` is not scalar-valued.
    """
    # parameter dimension
    n_par = len(x)

    nabla_1 = []
    for ix in range(n_par):
        delta_val = delta_vec[ix]
        delta = delta_val * unit_vec(dim=n_par, ix=ix)

        fp = fun(x + delta / 2)
        fm = fun(x - delta / 2)

        nabla_1.append((fp - fm) / delta_val)

    return np.array(nabla_1)


def unit_vec(dim: int, ix: int) -> np.ndarray:
    """Unit vector of dimension `dim` at coordinate `ix`.

    Parameters
    ----------
    dim: Vector dimension.
    ix: Index to contain the unit value.

    Returns
    -------
    vector: The unit vector.
    """
    vector = np.zeros(shape=dim)
    vector[ix] = 1
    return vector
