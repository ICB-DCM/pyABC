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
        size: tuple = None):
    # preprocess input
    histories, labels = to_lists_or_default(histories, labels)
    if scale is None:
        scale = 'lin'
    
    fig, ax = plt.subplots()

    # extract epsilons
    eps = []
    for history in histories:
        # note: first entry is from calibration, thus must be discarded
        eps.append(np.array(history.get_all_populations()['epsilon'][1:]))

    # scale
    if scale == 'log':
        eps = [np.log(ep) for ep in eps]
    elif scale == 'log10':
        eps = [np.log10(ep) for ep in eps]
    elif scale != 'lin':
        raise ValueError(f"Scale {scale} must be one of lin, log, log10.")

    # plot
    for ep, label in zip(eps, labels):
        ax.plot(ep, 'x-', label=label)

    ax.set_xlabel("Population index")
    if scale == 'log':
        ylabel = "Log(Epsilon)"
    elif scale == 'log10':
        ylabel = "Log10(Epsilon)"
    else:
        ylabel = "Epsilon"
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title("Epsilon values")
    # enforce integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # set size
    if size is not None:
        fig.set_size_inches(size)

    fig.tight_layout()

    return ax
