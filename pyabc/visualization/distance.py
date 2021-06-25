"""Visualization of distance functions."""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.axes
import numpy as np
from typing import List, Tuple, Union

from ..storage import load_dict_from_json

from .util import to_lists, get_labels


def plot_distance_weights(
    log_files: Union[List[str], str],
    ts: Union[List[int], List[str], int, str] = "last",
    labels: Union[List[str], str] = None,
    keys_as_labels: bool = True,
    xticklabel_rotation: float = 0,
    normalize: bool = True,
    size: Tuple[float, float] = None,
    ax: mpl.axes.Axes = None,
) -> mpl.axes.Axes:
    """Plot distance weights, one curve per argument.

    Assumes that the weights to be plotted from each file and timepoint have
    the same keys.

    Parameters
    ----------
    log_files:
        The weight log files, as passed to the distance functions, e.g. as
        "scale_log_file" or "info_log_file".
    ts:
        Time points to plot. Defaults to the last entry in each file.
    labels:
        A label for each log file.
    keys_as_labels:
        Whether to use the summary statistic keys as x tick labels.
    xticklabel_rotation:
        Angle by which to rotate x tick labels.
    normalize:
        Whether to normalize the weights to sum 1.
    size:
        Figure size in inches, (width, height).
    ax:
        Axis object to use.

    Returns
    -------
    The used axis object.
    """
    log_files, ts = to_lists(log_files, ts)
    labels = get_labels(labels, len(log_files))

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    n_run = len(log_files)

    # to remember the keys order
    keys = None

    # add a line per file
    for log_file, t, label in zip(log_files, ts, labels):
        weights = load_dict_from_json(log_file)
        if t == "last":
            t = max(weights.keys())
        weights = weights[t]
        if keys is None:
            keys = list(weights.keys())
        weights = np.array([weights[key] for key in keys])
        if normalize:
            weights /= weights.sum()
        ax.plot(weights, 'x-', label=label)

    # add labels
    if n_run > 1:
        ax.legend()
    ax.set_xticks(np.arange(len(keys)))
    if keys_as_labels:
        ax.set_xticklabels(keys, rotation=xticklabel_rotation)
    ax.set_xlabel("Summary statistic")
    ax.set_ylabel("Weight")

    if size is not None:
        fig.set_size_inches(size)

    return ax
