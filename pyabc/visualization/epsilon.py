"""Epsilon threshold plots"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.axes
from matplotlib.ticker import MaxNLocator
from typing import Union, List
import numpy as np

from ..storage import History
from .util import to_lists, get_labels


def plot_epsilons(
        histories: Union[List, History],
        labels: Union[List, str] = None,
        colors: List = None,
        yscale: str = 'log',
        title: str = "Epsilon values",
        size: tuple = None,
        ax: mpl.axes.Axes = None) -> mpl.axes.Axes:
    """
    Plot epsilon trajectory.

    Parameters
    ----------
    histories:
        The histories to plot from. History ids must be set correctly.
    labels:
        Labels corresponding to the histories. If None are provided,
        indices are used as labels.
    colors:
        Colors to use for the lines. If None, then the matplotlib
        default values are used.
    yscale:
        Scaling to apply to the y-axis. Use matplotlib's notation.
    title:
        Title for the plot.
    size:
        The size of the plot in inches.
    ax:
        The axis object to use. A new one is created if None.

    Returns
    -------
    ax: Axis of the generated plot.
    """
    # preprocess input
    histories = to_lists(histories)
    labels = get_labels(labels, len(histories))
    if colors is None:
        colors = [None for _ in range(len(histories))]

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # extract epsilons
    eps = []
    for history in histories:
        # note: first entry is from calibration and thus translates to inf,
        # thus must be discarded
        eps.append(np.array(history.get_all_populations()['epsilon'][1:]))

    # plot
    for ep, label, color in zip(eps, labels, colors):
        ax.plot(ep, 'x-', label=label, color=color)

    # format
    ax.set_xlabel("Population index")
    ax.set_ylabel("Epsilon")
    if any(lab is not None for lab in labels):
        ax.legend()
    ax.set_title(title)
    ax.set_yscale(yscale)
    # enforce integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # set size
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return ax
