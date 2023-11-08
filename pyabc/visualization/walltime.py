"""Walltime plots"""

import datetime
from typing import TYPE_CHECKING, Any, List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

if TYPE_CHECKING:
    import plotly.graph_objs as go

from ..storage import History
from .util import get_labels, to_lists

SECOND = 's'
MINUTE = 'm'
HOUR = 'h'
DAY = 'd'
TIME_UNITS = [SECOND, MINUTE, HOUR, DAY]


def _prepare_plot_total_walltime(
    histories: Union[List[History], History],
    labels: Union[List, str],
    unit: str,
):
    # preprocess input
    histories = to_lists(histories)
    labels = get_labels(labels, len(histories))
    n_run = len(histories)

    # check time unit
    if unit not in TIME_UNITS:
        raise AssertionError(f"`unit` must be in {TIME_UNITS}")

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
        walltimes /= 60 * 60
    elif unit == DAY:
        walltimes /= 60 * 60 * 24

    return labels, n_run, walltimes


def plot_total_walltime(
    histories: Union[List[History], History],
    labels: Union[List, str] = None,
    unit: str = 's',
    rotation: int = 0,
    title: str = "Total walltimes",
    size: tuple = None,
    ax: mpl.axes.Axes = None,
) -> mpl.axes.Axes:
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

    Returns
    -------
    A reference to the axis of the generated plot.
    """
    # preprocess input
    labels, n_run, walltimes = _prepare_plot_total_walltime(
        histories=histories, labels=labels, unit=unit
    )

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

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


def plot_total_walltime_plotly(
    histories: Union[List[History], History],
    labels: Union[List, str] = None,
    unit: str = 's',
    rotation: int = 0,
    title: str = "Total walltimes",
    size: tuple = None,
    fig: "go.Figure" = None,
) -> "go.Figure":
    """Plot total walltimes using plotly."""
    import plotly.graph_objects as go

    # preprocess input
    labels, n_run, walltimes = _prepare_plot_total_walltime(
        histories=histories, labels=labels, unit=unit
    )

    # create figure
    if fig is None:
        fig = go.Figure()

    # plot bars
    fig.add_trace(go.Bar(x=np.arange(n_run), y=walltimes))

    # prettify plot
    fig.update_layout(
        xaxis={
            'tickmode': 'array',
            'tickvals': np.arange(n_run),
            'ticktext': labels,
            'tickangle': rotation,
        },
        title=title,
        xaxis_title="Run",
        yaxis_title=f"Time [{unit}]",
    )
    if size is not None:
        fig.update_layout(width=size[0], height=size[1])

    return fig


def _prepare_walltime(
    histories: Union[List[History], History],
    show_calibration: bool,
):
    # preprocess input
    histories = to_lists(histories)

    # show calibration if that makes sense
    if show_calibration is None:
        show_calibration = any(
            h.get_all_populations().samples[0] > 0 for h in histories
        )

    # extract start times and end times
    start_times = []
    end_times = []
    for h in histories:
        # start time
        start_times.append(h.get_abc().start_time)
        # end times
        end_times.append(h.get_all_populations().population_end_time)

    return start_times, end_times, show_calibration


def plot_walltime(
    histories: Union[List[History], History],
    labels: Union[List, str] = None,
    show_calibration: bool = None,
    unit: str = 's',
    rotation: int = 0,
    title: str = "Walltime by generation",
    size: tuple = None,
    ax: mpl.axes.Axes = None,
) -> mpl.axes.Axes:
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
    ax:
        A reference to the axis of the generated plot.
    """
    # preprocess input
    start_times, end_times, show_calibration = _prepare_walltime(
        histories=histories, show_calibration=show_calibration
    )

    return plot_walltime_lowlevel(
        end_times=end_times,
        start_times=start_times,
        labels=labels,
        show_calibration=show_calibration,
        unit=unit,
        rotation=rotation,
        title=title,
        size=size,
        ax=ax,
    )


def plot_walltime_plotly(
    histories: Union[List[History], History],
    labels: Union[List, str] = None,
    show_calibration: bool = None,
    unit: str = 's',
    rotation: int = 0,
    title: str = "Walltime by generation",
    size: tuple = None,
    fig: "go.Figure" = None,
) -> "go.Figure":
    """Plot walltimes using plotly."""
    # preprocess input
    start_times, end_times, show_calibration = _prepare_walltime(
        histories=histories, show_calibration=show_calibration
    )

    return plot_walltime_lowlevel_plotly(
        end_times=end_times,
        start_times=start_times,
        labels=labels,
        show_calibration=show_calibration,
        unit=unit,
        rotation=rotation,
        title=title,
        size=size,
        fig=fig,
    )


def _prepare_plot_walltime_lowlevel(
    end_times: List,
    start_times: Union[List, None] = None,
    labels: Union[List, str] = None,
    show_calibration: bool = None,
    unit: str = 's',
):
    # preprocess input
    end_times = to_lists(end_times)
    labels = get_labels(labels, len(end_times))
    n_run = len(end_times)

    # check start times
    if start_times is None:
        if show_calibration:
            raise AssertionError(
                "To plot the calibration iteration, start times are needed."
            )
        # fill in dummy times which will not be used anyhow
        start_times = [datetime.datetime.now() for _ in range(n_run)]

    # check time unit
    if unit not in TIME_UNITS:
        raise AssertionError(f"`unit` must be in {TIME_UNITS}")

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
        matrix[: len(wt), i_run] = wt

    if not show_calibration:
        matrix = matrix[1:, :]

    # apply time unit
    if unit == MINUTE:
        matrix /= 60
    elif unit == HOUR:
        matrix /= 60 * 60
    elif unit == DAY:
        matrix /= 60 * 60 * 24

    return matrix, labels, n_run


def plot_walltime_lowlevel(
    end_times: List,
    start_times: Union[List, None] = None,
    labels: Union[List, str] = None,
    show_calibration: bool = None,
    unit: str = 's',
    rotation: int = 0,
    title: str = "Walltime by generation",
    size: tuple = None,
    ax: mpl.axes.Axes = None,
) -> mpl.axes.Axes:
    """Low-level access to `plot_walltime`.

    Directly define `end_times` and `start_times`.
    """
    # preprocess input
    matrix, labels, n_run = _prepare_plot_walltime_lowlevel(
        end_times=end_times,
        start_times=start_times,
        labels=labels,
        show_calibration=show_calibration,
        unit=unit,
    )

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # plot bars
    for i_pop in reversed(range(matrix.shape[0])):
        pop_ix = i_pop - 1
        if not show_calibration:
            pop_ix = i_pop
        ax.bar(
            x=np.arange(n_run),
            height=matrix[i_pop, :],
            bottom=np.sum(matrix[:i_pop, :], axis=0),
            label=f"Generation {pop_ix}",
        )

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


def plot_walltime_lowlevel_plotly(
    end_times: List,
    start_times: Union[List, None] = None,
    labels: Union[List, str] = None,
    show_calibration: bool = None,
    unit: str = 's',
    rotation: int = 0,
    title: str = "Walltime by generation",
    size: tuple = None,
    fig: "go.Figure" = None,
) -> "go.Figure":
    """Low-level access to `plot_walltime_plotly`."""
    import plotly.graph_objects as go

    # preprocess input
    matrix, labels, n_run = _prepare_plot_walltime_lowlevel(
        end_times=end_times,
        start_times=start_times,
        labels=labels,
        show_calibration=show_calibration,
        unit=unit,
    )

    # create figure
    if fig is None:
        fig = go.Figure()

    # plot bars
    for i_pop in reversed(range(matrix.shape[0])):
        pop_ix = i_pop - 1
        if not show_calibration:
            pop_ix = i_pop
        fig.add_trace(
            go.Bar(
                x=np.arange(n_run),
                y=matrix[i_pop, :],
                name=f"Generation {pop_ix}",
                offsetgroup=0,
                base=np.sum(matrix[:i_pop, :], axis=0),
            )
        )

    # prettify plot
    fig.update_layout(
        xaxis_title="Run",
        yaxis_title=f"Time [{unit}]",
        title=title,
        xaxis_tickvals=np.arange(n_run),
        xaxis_ticktext=labels,
        xaxis_tickangle=rotation,
        legend_title="Generation",
    )

    if size is not None:
        fig.update_layout(width=size[0], height=size[1])

    return fig


def _prepare_plot_eps_walltime(
    histories: Union[List[History], History],
):
    # preprocess input
    histories = to_lists(histories)

    # extract end times and epsilons
    end_times = []
    eps = []
    for h in histories:
        # end times
        end_times.append(h.get_all_populations().population_end_time)
        eps.append(h.get_all_populations().epsilon.to_numpy())

    return end_times, eps


def plot_eps_walltime(
    histories: Union[List[History], History],
    labels: Union[List, str] = None,
    colors: List[Any] = None,
    group_by_label: bool = True,
    indicate_end: bool = True,
    unit: str = 's',
    xscale: str = 'linear',
    yscale: str = 'log',
    title: str = "Epsilon over walltime",
    size: tuple = None,
    ax: mpl.axes.Axes = None,
) -> mpl.axes.Axes:
    """Plot epsilon values (y-axis) over the walltime (x-axis), iterating over
    the generations.

    Parameters
    ----------
    histories:
        The histories to plot from. History ids must be set correctly.
    labels:
        Labels corresponding to the histories. If None are provided,
        indices are used as labels.
    colors:
        One color for each history.
    group_by_label:
        Whether to group colors (unless explicitly provided) and legends by
        label.
    indicate_end:
        Whether to indicate the final time by a line.
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
    A reference of the plot object.
    """
    # preprocess input
    end_times, eps = _prepare_plot_eps_walltime(histories=histories)

    return plot_eps_walltime_lowlevel(
        end_times=end_times,
        eps=eps,
        labels=labels,
        colors=colors,
        group_by_label=group_by_label,
        indicate_end=indicate_end,
        unit=unit,
        xscale=xscale,
        yscale=yscale,
        title=title,
        size=size,
        ax=ax,
    )


def plot_eps_walltime_plotly(
    histories: Union[List[History], History],
    labels: Union[List, str] = None,
    colors: List[Any] = None,
    group_by_label: bool = True,
    indicate_end: bool = True,
    unit: str = 's',
    xscale: str = 'linear',
    yscale: str = 'log',
    title: str = "Epsilon over walltime",
    size: tuple = None,
    fig: "go.Figure" = None,
) -> "go.Figure":
    """Plot epsilon values over walltime using plotly."""
    # preprocess input
    end_times, eps = _prepare_plot_eps_walltime(histories=histories)

    return plot_eps_walltime_lowlevel_plotly(
        end_times=end_times,
        eps=eps,
        labels=labels,
        colors=colors,
        group_by_label=group_by_label,
        indicate_end=indicate_end,
        unit=unit,
        xscale=xscale,
        yscale=yscale,
        title=title,
        size=size,
        fig=fig,
    )


def _prepare_plot_eps_walltime_lowlevel(
    end_times: List,
    eps: List,
    labels: Union[List, str],
    colors: List[Any],
    group_by_label: bool,
    unit: str,
):
    # preprocess input
    end_times = to_lists(end_times)
    labels = get_labels(labels, len(end_times))
    n_run = len(end_times)

    if group_by_label:
        if colors is None:
            colors = []
            color_ix = -1
            for ix, label in enumerate(labels):
                if label not in labels[:ix]:
                    color_ix += 1
                colors.append(f"C{color_ix}")

        labels = [
            x if x not in labels[:ix] else None for ix, x in enumerate(labels)
        ]
    if colors is None:
        colors = [None] * n_run

    # check time unit
    if unit not in TIME_UNITS:
        raise AssertionError(f"`unit` must be in {TIME_UNITS}")

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

    return walltimes, eps, labels, colors, n_run


def plot_eps_walltime_lowlevel(
    end_times: List,
    eps: List,
    labels: Union[List, str] = None,
    colors: List[Any] = None,
    group_by_label: bool = True,
    indicate_end: bool = True,
    unit: str = 's',
    xscale: str = 'linear',
    yscale: str = 'log',
    title: str = "Epsilon over walltime",
    size: tuple = None,
    ax: mpl.axes.Axes = None,
) -> mpl.axes.Axes:
    """Low-level access to `plot_eps_walltime`.
    Directly define `end_times` and `eps`. Note that both should be arrays of
    the same length and at the beginning include a value for the calibration
    iteration. This is just what `pyabc.History.get_all_populations()` returns.
    The first time is used as the base time differences to which are plotted.
    The first epsilon is ignored.
    """
    # preprocess input
    (
        walltimes,
        eps,
        labels,
        colors,
        n_run,
    ) = _prepare_plot_eps_walltime_lowlevel(
        end_times=end_times,
        eps=eps,
        labels=labels,
        colors=colors,
        group_by_label=group_by_label,
        unit=unit,
    )

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for wt, ep, label, color in zip(walltimes, eps, labels, colors):
        wt = np.asarray(wt)
        # apply time unit
        if unit == MINUTE:
            wt /= 60
        elif unit == HOUR:
            wt /= 60 * 60
        elif unit == DAY:
            wt /= 60 * 60 * 24
        # plot
        ax.plot(wt, ep, label=label, marker='o', color=color)
        if indicate_end:
            ax.axvline(wt[-1], linestyle='dashed', color=color)

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


def plot_eps_walltime_lowlevel_plotly(
    end_times: List,
    eps: List,
    labels: Union[List, str] = None,
    colors: List[Any] = None,
    group_by_label: bool = True,
    indicate_end: bool = True,
    unit: str = 's',
    xscale: str = 'linear',
    yscale: str = 'log',
    title: str = "Epsilon over walltime",
    size: tuple = None,
    fig: "go.Figure" = None,
) -> "go.Figure":
    """Plot epsilon values over walltime using plotly."""
    import plotly.graph_objects as go

    # preprocess input
    walltimes, eps, labels, colors, _ = _prepare_plot_eps_walltime_lowlevel(
        end_times=end_times,
        eps=eps,
        labels=labels,
        colors=colors,
        group_by_label=group_by_label,
        unit=unit,
    )

    # create figure
    if fig is None:
        fig = go.Figure()

    for wt, ep, label in zip(walltimes, eps, labels):
        wt = np.asarray(wt)
        # apply time unit
        if unit == MINUTE:
            wt /= 60
        elif unit == HOUR:
            wt /= 60 * 60
        elif unit == DAY:
            wt /= 60 * 60 * 24
        # plot
        fig.add_trace(
            go.Scatter(
                x=wt,
                y=ep,
                name=label,
                mode='lines+markers',
            )
        )
        if indicate_end:
            # add a vertical line from minimum to maximum value
            fig.add_shape(
                type="line",
                x0=wt[-1],
                y0=ep[-1],
                x1=wt[-1],
                y1=ep[0],
                line={
                    'width': 1,
                    'dash': "dash",
                },
            )

    # prettify plot
    fig.update_layout(
        title=title,
        xaxis_title=f"Time [{unit}]",
        yaxis_title="Epsilon",
        xaxis_type=xscale,
        yaxis_type=yscale,
    )

    if size is not None:
        fig.update_layout(width=size[0], height=size[1])

    return fig
