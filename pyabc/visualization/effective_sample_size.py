import matplotlib.pyplot as plt
from typing import List, Union
from matplotlib.ticker import MaxNLocator

from ..weighted_statistics import effective_sample_size
from ..storage import History
from .util import to_lists_or_default


def plot_effective_sample_sizes(
        histories: Union[List, History],
        labels: Union[List, str] = None,
        rotation: int = 0,
        title: str = "Effective sample size",
        size: tuple = None):
    """
    Plot effective sample sizes over all iterations.

    Parameters
    ----------

    histories: Union[List, History]
        The histories to plot from. History ids must be set correctly.
    labels: Union[List ,str], optional
        Labels corresponding to the histories. If None are provided,
        indices are used as labels.
    rotation: int, optional (default = 0)
        Rotation to apply to the plot's x tick labels. For longer labels,
        a tilting of 45 or even 90 can be preferable.
    title: str, optional (default = "Total required samples")
        Title for the plot.
    size: tuple of float, optional
        The size of the plot in inches.

    Returns
    -------

    ax: Axis of the generated plot.
    """
    # preprocess input
    histories, labels = to_lists_or_default(histories, labels)

    # create figure
    fig, ax = plt.subplots()

    # extract effective sample sizes
    essss = []  # :)
    for history in histories:
        esss = []
        for t in range(0, history.max_t + 1):
            # we need the weights not normalized to 1 for each model
            w = history.get_weighted_distances(t=t)['w']
            ess = effective_sample_size(w)
            esss.append(ess)
        essss.append(esss)

    # plot
    for esss, label in zip(essss, labels):
        ax.plot(range(0, len(esss)), esss, 'x-', label=label)

    # format
    ax.set_xlabel("Population index")
    ax.set_ylabel("ESS")
    ax.legend()
    ax.set_title(title)
    # enforce integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # set size
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return ax
