"""Visualization of distance functions."""

from typing import Any, List, Tuple, Union

import matplotlib as mpl
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from ..storage import load_dict_from_json
from .util import get_labels, to_lists


def plot_distance_weights(
    log_files: Union[List[str], str],
    ts: Union[List[int], List[str], int, str] = "last",
    labels: Union[List[str], str] = None,
    colors: Union[List[Any], Any] = None,
    linestyles: Union[List[str], str] = None,
    keys_as_labels: bool = True,
    keys: List[str] = None,
    xticklabel_rotation: float = 0,
    normalize: bool = True,
    size: Tuple[float, float] = None,
    xlabel: str = "Summary statistic",
    ylabel: str = "Weight",
    title: str = None,
    ax: mpl.axes.Axes = None,
    **kwargs,
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
    colors:
        A color for each log file.
    linestyles:
        Linestyles to apply.
    keys_as_labels:
        Whether to use the summary statistic keys as x tick labels.
    keys:
        Data keys to plot.
    xticklabel_rotation:
        Angle by which to rotate x tick labels.
    normalize:
        Whether to normalize the weights to sum 1.
    size:
        Figure size in inches, (width, height).
    xlabel:
        x-axis label.
    ylabel:
        y-axis label.
    title:
        Plot title.
    ax:
        Axis object to use.
    **kwargs:
        Additional keyword arguments are passed on to `plt.plot()`
        when plotting lines.

    Returns
    -------
    The used axis object.
    """
    log_files, ts, colors, linestyles = to_lists(
        log_files, ts, colors, linestyles
    )
    labels = get_labels(labels, len(log_files))

    # default keyword arguments
    if "marker" not in kwargs:
        kwargs["marker"] = "x"

    n_run = len(log_files)

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # add a line per file
    for log_file, t, label, color, linestyle in zip(
        log_files, ts, labels, colors, linestyles
    ):
        weights = load_dict_from_json(log_file)
        if t == "last":
            t = max(weights.keys())
        weights = weights[t]
        if keys is None:
            keys = list(weights.keys())
        weights = np.array([weights[key] for key in keys])
        if normalize:
            weights /= weights.sum()
        ax.plot(
            weights,
            label=label,
            color=color,
            linestyle=linestyle,
            **kwargs,
        )

    # add labels
    if n_run > 1:
        ax.legend()

    # x axis ticks
    if keys_as_labels:
        ax.set_xticks(np.arange(len(keys)))
        ax.set_xticklabels(keys, rotation=xticklabel_rotation)
    else:
        # enforce integer labels
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if size is not None:
        fig.set_size_inches(size)

    return ax
