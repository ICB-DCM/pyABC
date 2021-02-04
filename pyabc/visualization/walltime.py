"""Walltime plots"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.axes
from matplotlib.ticker import MaxNLocator
import datetime
from typing import List, Union

from ..storage import History
from .util import to_lists, get_labels

SECOND = 's'
MINUTE = 'm'
HOUR = 'h'
DAY = 'd'
TIME_UNITS = [SECOND, MINUTE, HOUR, DAY]


def plot_total_walltime(
        histories: Union[List[History], History],
        labels: Union[List, str] = None,
        unit: str = 's',
        rotation: int = 0,
        title: str = "Total walltimes",
        size: tuple = None,
        ax: mpl.axes.Axes = None) -> mpl.axes.Axes:
    """Plot total walltimes, for each history one single-color bar.

    Parameters
    ----------
    histories:
        The histories to plot from. History ids must be set correctly.
    labels:
        Labels corresponding to the histories. If None are provided,
        indices are used as labels.
    unit:
        Time unit to use ('s', 'm', 'h', 'd' as seconds, minutes, hours, days).
    rotation:
        Rotation to apply to the plot's x tick labels. For longer labels,
        a tilting of 45 or even 90 can be preferable.
    title:
        Title for the plot.
    size: tuple of float, optional
        The size of the plot in inches.
    ax: matplotlib.axes.Axes, optional
        The axis object to use.

    Returns
    -------
    ax: Axis of the generated plot.
    """
    # preprocess input
    histories = to_lists(histories)
    labels = get_labels(labels, len(histories))
    n_run = len(histories)

    # check time unit
    if unit not in TIME_UNITS:
        raise AssertionError(f"`unit` must be in {TIME_UNITS}")

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # extract total walltimes
    walltimes = []
    for h in histories:
        abc = h.get_abc()
        walltimes.append((abc.end_time - abc.start_time).total_seconds())
    walltimes = np.asarray(walltimes)

    # apply time unit
    if unit == MINUTE:
        walltimes /= 60
    elif unit == HOUR:
        walltimes /= (60*60)
    elif unit == DAY:
        walltimes /= (60*60*24)

    # plot bars
    ax.bar(x=np.arange(n_run), height=walltimes, label=labels)

    # prettify plot
    ax.set_xticks(np.arange(n_run))
    ax.set_xticklabels(labels, rotation=rotation)
    ax.set_title(title)
    ax.set_xlabel("Run")
    ax.set_ylabel(f"Time [{unit}]")
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return ax


def plot_walltime(
        histories: Union[List[History], History],
        labels: Union[List, str] = None,
        show_calibration: bool = None,
        unit: str = 's',
        rotation: int = 0,
        title: str = "Walltime by generation",
        size: tuple = None,
        ax: mpl.axes.Axes = None) -> mpl.axes.Axes:
    """Plot walltimes, with different colors indicating different iterations.

    Parameters
    ----------
    histories:
        The histories to plot from. History ids must be set correctly.
    labels:
        Labels corresponding to the histories. If None are provided,
        indices are used as labels.
    show_calibration:
        Whether to show the calibration iteration (-1). Defaults to whether
        there are samples in the calibration iteration.
    unit:
        Time unit to use ('s', 'm', 'h', 'd' as seconds, minutes, hours, days).
    rotation:
        Rotation to apply to the plot's x tick labels. For longer labels,
        a tilting of 45 or even 90 can be preferable.
    title:
        Title for the plot.
    size: tuple of float, optional
        The size of the plot in inches.
    ax: matplotlib.axes.Axes, optional
        The axis object to use.

    Returns
    -------
    ax: Axis of the generated plot.
    """
    # preprocess input
    histories = to_lists(histories)

    # show calibration if that makes sense
    if show_calibration is None:
        show_calibration = any(
            h.get_all_populations().samples[0] > 0 for h in histories)

    # extract start times and end times
    start_times = []
    end_times = []
    for h in histories:
        # start time
        start_times.append(h.get_abc().start_time)
        # end times
        end_times.append(h.get_all_populations().population_end_time)

    return plot_walltime_lowlevel(
        end_times=end_times, start_times=start_times, labels=labels,
        show_calibration=show_calibration, unit=unit, rotation=rotation,
        title=title, size=size, ax=ax)


def plot_walltime_lowlevel(
        end_times: List,
        start_times: Union[List, None] = None,
        labels: Union[List, str] = None,
        show_calibration: bool = None,
        unit: str = 's',
        rotation: int = 0,
        title: str = "Walltime by generation",
        size: tuple = None,
        ax: mpl.axes.Axes = None) -> mpl.axes.Axes:
    """Low-level access to `plot_walltime`.
    Directly define `end_times` and `start_times`."""
    # preprocess input
    end_times = to_lists(end_times)
    labels = get_labels(labels, len(end_times))
    n_run = len(end_times)

    # check start times
    if start_times is None:
        if show_calibration:
            raise AssertionError(
                "To plot the calibration iteration, start times are needed.")
        # fill in dummy times which will not be used anyhow
        start_times = [datetime.datetime.now() for _ in range(n_run)]

    # check time unit
    if unit not in TIME_UNITS:
        raise AssertionError(f"`unit` must be in {TIME_UNITS}")

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # extract relative walltimes
    walltimes = []
    for start_t, end_ts in zip(start_times, end_times):
        times = [start_t, *end_ts]
        # compute stacked differences
        diffs = [end - start for start, end in zip(times[:-1], times[1:])]
        # as seconds
        diffs = [diff.total_seconds() for diff in diffs]
        # append
        walltimes.append(diffs)
    walltimes = np.asarray(walltimes)

    # create matrix
    n_pop = max(len(wt) for wt in walltimes)
    matrix = np.zeros((n_pop, n_run))
    for i_run, wt in enumerate(walltimes):
        matrix[:len(wt), i_run] = wt

    if not show_calibration:
        matrix = matrix[1:, :]

    # apply time unit
    if unit == MINUTE:
        matrix /= 60
    elif unit == HOUR:
        matrix /= (60*60)
    elif unit == DAY:
        matrix /= (60*60*24)

    # plot bars
    for i_pop in reversed(range(matrix.shape[0])):
        pop_ix = i_pop - 1
        if not show_calibration:
            pop_ix = i_pop
        ax.bar(x=np.arange(n_run),
               height=matrix[i_pop, :],
               bottom=np.sum(matrix[:i_pop, :], axis=0),
               label=f"Generation {pop_ix}")

    # prettify plot
    ax.set_xticks(np.arange(n_run))
    ax.set_xticklabels(labels, rotation=rotation)
    ax.set_title(title)
    ax.set_xlabel("Run")
    ax.set_ylabel(f"Time [{unit}]")
    ax.legend()
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return ax


def plot_eps_walltime(
        histories: Union[List[History], History],
        labels: Union[List, str] = None,
        unit: str = 's',
        xscale: str = 'linear',
        yscale: str = 'log',
        title: str = "Epsilon over walltime",
        size: tuple = None,
        ax: mpl.axes.Axes = None) -> mpl.axes.Axes:
    """Plot epsilon values (y-axis) over the walltime (x-axis), iterating over
    the generations.

    Parameters
    ----------
    histories:
        The histories to plot from. History ids must be set correctly.
    labels:
        Labels corresponding to the histories. If None are provided,
        indices are used as labels.
    unit:
        Time unit to use ('s', 'm', 'h', 'd' as seconds, minutes, hours, days).
    xscale:
        Scale of the x-axis. Use matplotlib's notation.
    yscale:
        Scale of the y-axis. Use matplotlib's notation.
    title:
        Title for the plot.
    size: tuple of float, optional
        The size of the plot in inches.
    ax: matplotlib.axes.Axes, optional
        The axis object to use.

    Returns
    -------
    ax: Axis of the generated plot.
    """
    # preprocess input
    histories = to_lists(histories)

    # extract end times and epsilons
    end_times = []
    eps = []
    for h in histories:
        # end times
        end_times.append(h.get_all_populations().population_end_time)
        eps.append(h.get_all_populations().epsilon)

    return plot_eps_walltime_lowlevel(
        end_times=end_times, eps=eps, labels=labels, unit=unit,
        xscale=xscale, yscale=yscale, title=title, size=size, ax=ax)


def plot_eps_walltime_lowlevel(
        end_times: List,
        eps: List,
        labels: Union[List, str] = None,
        unit: str = 's',
        xscale: str = 'linear',
        yscale: str = 'log',
        title: str = "Epsilon over walltime",
        size: tuple = None,
        ax: mpl.axes.Axes = None) -> mpl.axes.Axes:
    """Low-level access to `plot_eps_walltime`.
    Directly define `end_times` and `eps`. Note that both should be arrays of
    the same length and at the beginning include a value for the calibration
    iteration. This is just what `pyabc.History.get_all_populations()` returns.
    The first time is used as the base time differences to which are plotted.
    The first epsilon is ignored.
    """
    # preprocess input
    end_times = to_lists(end_times)
    labels = get_labels(labels, len(end_times))
    n_run = len(end_times)

    # check time unit
    if unit not in TIME_UNITS:
        raise AssertionError(f"`unit` must be in {TIME_UNITS}")

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # extract relative walltimes
    walltimes = []
    for end_ts in end_times:
        # compute differences to base
        diffs = end_ts[1:] - end_ts[0]
        # as seconds
        diffs = [diff.total_seconds() for diff in diffs]
        # append
        walltimes.append(diffs)

    # disregard calibration epsilon (inf)
    eps = [ep[1:] for ep in eps]

    for wt, ep, label in zip(walltimes, eps, labels):
        wt = np.asarray(wt)
        # apply time unit
        if unit == MINUTE:
            wt /= 60
        elif unit == HOUR:
            wt /= (60 * 60)
        elif unit == DAY:
            wt /= (60 * 60 * 24)
        # plot
        ax.plot(wt, ep, label=label, marker='o')

    # prettify plot
    if n_run > 1:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(f"Time [{unit}]")
    ax.set_ylabel("Epsilon")
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    # enforce integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return ax
