"""Effective sample size plots"""

from typing import TYPE_CHECKING, List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from ..storage import History
from ..weighted_statistics import effective_sample_size
from .util import get_labels, to_lists

if TYPE_CHECKING:
    import plotly.graph_objs as go


def _prepare_plot_effective_sample_sizes(
    histories: Union[List, History],
    labels: Union[List, str],
    colors: List,
    relative: bool,
):
    # preprocess input
    histories = to_lists(histories)
    labels = get_labels(labels, len(histories))
    if colors is None:
        colors = [None for _ in range(len(histories))]

    # extract effective sample sizes
    essss = []  # :)
    for history in histories:
        esss = []
        for t in range(0, history.max_t + 1):
            # we need the weights not normalized to 1 for each model
            w = history.get_weighted_distances(t=t)['w']
            ess = effective_sample_size(w)
            if relative:
                ess /= len(w)
            esss.append(ess)
        essss.append(esss)

    return labels, colors, essss


def plot_effective_sample_sizes(
    histories: Union[List, History],
    labels: Union[List, str] = None,
    rotation: int = 0,
    title: str = "Effective sample size",
    relative: bool = False,
    colors: List = None,
    size: tuple = None,
    ax: mpl.axes.Axes = None,
) -> mpl.axes.Axes:
    """
    Plot effective sample sizes over all iterations.

    Parameters
    ----------
    histories:
        The histories to plot from. History ids must be set correctly.
    labels:
        Labels corresponding to the histories. If None are provided,
        indices are used as labels.
    rotation:
        Rotation to apply to the plot's x tick labels. For longer labels,
        a tilting of 45 or even 90 can be preferable.
    title:
        Title for the plot.
    relative:
        Whether to show relative sizes (to 1) or w.r.t. the real number
        of particles.
    colors:
        Colors to use for the lines. If None, then the matplotlib
        default values are used.
    size:
        The size of the plot in inches.
    ax:
        The matplotlib axes object to use. If None, a new figure is
        created.

    Returns
    -------
    ax:
        The matplotlib axes object.
    """
    # prepare data
    labels, colors, essss = _prepare_plot_effective_sample_sizes(
        histories, labels, colors, relative
    )

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # plot
    for esss, label, color in zip(essss, labels, colors):
        ax.plot(range(0, len(esss)), esss, 'x-', label=label, color=color)

    # format
    ax.set_xlabel("Population index")
    ax.set_ylabel("ESS")
    if any(lab is not None for lab in labels):
        ax.legend()
    ax.set_title(title)
    # enforce integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # rotate x tick labels
    plt.setp(ax.get_xticklabels(), rotation=rotation)

    # set size
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return ax


def plot_effective_sample_sizes_plotly(
    histories: Union[List, History],
    labels: Union[List, str] = None,
    rotation: int = 0,
    title: str = "Effective sample size",
    relative: bool = False,
    colors: List = None,
    size: tuple = None,
    fig: "go.Figure" = None,
) -> "go.Figure":
    """Plot effective sample sizes using plotly."""
    import plotly.graph_objects as go

    # prepare data
    labels, colors, essss = _prepare_plot_effective_sample_sizes(
        histories, labels, colors, relative
    )

    # create figure
    if fig is None:
        fig = go.Figure()

    # plot
    for esss, label, color in zip(essss, labels, colors):
        fig.add_trace(
            go.Scatter(
                x=list(range(0, len(esss))),
                y=esss,
                mode="lines+markers",
                name=label,
                marker={'color': color},
            )
        )

    # format
    fig.update_layout(
        xaxis_title="Population index",
        yaxis_title="ESS",
        title=title,
        xaxis={'tickmode': "linear"},
    )
    # rotate x tick labels
    fig.update_xaxes(tickangle=rotation)
    # set size
    if size is not None:
        fig.update_layout(width=size[0], height=size[1])

    return fig
