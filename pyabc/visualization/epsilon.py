import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Union, List
import numpy as np

from ..storage import History
from .util import to_lists_or_default


def plot_epsilons(
        histories: Union[List, History],
        labels: Union[List, str] = None,
        scale: str = None,
        title: str = "Epsilon values",
        size: tuple = None):
    """
    Plot epsilon trajectory.

    Parameters
    ----------

    histories: Union[List, History]
        The histories to plot from. History ids must be set correctly.
    labels: Union[List ,str], optional
        Labels corresponding to the histories. If None are provided,
        indices are used as labels.
    scale: str, optional (default='lin')
        Scaling to apply to the y axis.
        Must be one of 'lin', 'log', 'log10'.
    title: str, optional (default = "Epsilon values")
        Title for the plot.
    size: tuple of float, optional
        The size of the plot in inches.

    Returns
    -------

    ax: Axis of the generated plot.
    """
    # preprocess input
    histories, labels = to_lists_or_default(histories, labels)
    if scale is None:
        scale = 'lin'

    fig, ax = plt.subplots()

    # extract epsilons
    eps = []
    for history in histories:
        # note: first entry is from calibration and thus translates to inf,
        # thus must be discarded
        eps.append(np.array(history.get_all_populations()['epsilon'][1:]))

    # scale
    eps = _apply_scale(eps, scale)

    # plot
    for ep, label in zip(eps, labels):
        ax.plot(ep, 'x-', label=label)

    # format
    ax.set_xlabel("Population index")
    ax.set_ylabel(_get_ylabel(scale))
    ax.legend()
    ax.set_title(title)
    # enforce integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # set size
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return ax


def _apply_scale(eps, scale):
    """
    Apply the `scale` transformation to `eps`.
    """
    if scale == 'log':
        eps = [np.log(ep) for ep in eps]
    elif scale == 'log10':
        eps = [np.log10(ep) for ep in eps]
    elif scale != 'lin':
        raise ValueError(f"Scale {scale} must be one of lin, log, log10.")
    return eps


def _get_ylabel(scale):
    """
    Get corect y axis label.
    """
    if scale == 'log':
        ylabel = "Log(Epsilon)"
    elif scale == 'log10':
        ylabel = "Log10(Epsilon)"
    else:
        ylabel = "Epsilon"
    return ylabel
