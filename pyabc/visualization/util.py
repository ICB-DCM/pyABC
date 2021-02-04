"""Visualization util functions"""

from collections.abc import Sequence
import numpy as np


def to_lists(*args):
    """Convert to lists of the same length."""
    # tuple to array
    args = list(args)

    # entries to arrays
    for ix, arg in enumerate(args):
        if not isinstance(arg, list):
            args[ix] = [arg]

    # get length
    length = max(len(arg) for arg in args)

    # broadcast singulars
    for ix, arg in enumerate(args):
        if len(arg) == 1 and len(arg) != length:
            args[ix] = [arg[0]] * length

    # check length consistency
    if any(len(arg) != length for arg in args):
        raise AssertionError(f"The argument lengths are inconsistent: {args}")

    if len(args) == 1:
        return args[0]
    return args


def get_labels(labels, n: int, default_label: str = "Run"):
    """Create list of length `n` labels, using `default_label` if the only
    `label is None`."""
    # entry to array
    if not isinstance(labels, list):
        labels = [labels]

    # if the length is 1, add identifiers
    if n != len(labels) == 1:
        label = labels[0]
        if label is None:
            label = default_label
        labels = [f"{label} {ix}" for ix in range(n)]

    # check length consistency
    if len(labels) != n:
        raise AssertionError("The number of labels does not fit")

    return labels


def format_plot_matrix(arr_ax: np.ndarray, par_names: Sequence):
    """Clear all labels and legends, and set the left-most and bottom-most
    labels to the parameter names.

    Parameters
    ----------
    arr_ax:
        Array of mpl.axes.Axes.
        Shape (n_par, n_par) where len(par_names) == n_par.
    par_names:
        Parameter names to be used as labels.
    """
    n_par = len(par_names)

    for i in range(0, n_par):
        for j in range(0, n_par):
            # clear labels
            arr_ax[i, j].set_xlabel("")
            arr_ax[i, j].set_ylabel("")

            # clear legends
            arr_ax[i, j].legend = None

            # remove spines
            arr_ax[i, j].spines['right'].set_visible(False)
            arr_ax[i, j].spines['top'].set_visible(False)

    # set left-most and bottom-most labels to parameter names
    for ax, label in zip(arr_ax[-1, :], par_names):
        ax.set_xlabel(label)
    for ax, label in zip(arr_ax[:, 0], par_names):
        ax.set_ylabel(label)
