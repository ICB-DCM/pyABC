import matplotlib.pyplot as plt
from typing import Union, List
import numpy as np

from ..storage import History
from .util import get_lists_or_default


def plot_epsilons(
        histories: Union[List, History],
        labels: Union[List, str] = None,
        scale: str = None):
    # preprocess input
    histories, labels = get_lists_or_default(histories, labels)
    if scale is None:
        scale = 'lin'
    
    fig, ax = plt.subplots()

    # extract epsilons
    eps = []
    for history in histories:
        eps.append(np.array(history.get_all_populations()['epsilon']))

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
    ax.set_legend("Epsilon values")

    return ax
