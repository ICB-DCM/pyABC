import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union

from ..storage import History
from .util import to_lists_or_default

def plot_sample_numbers(
        histories: Union[List, History],
        labels: Union[List, str] = None,
        rotation: int = 0):
    """
    Plot required numbers of samples over all iterations.
    """
    # preprocess input
    histories, labels = to_lists_or_default(histories, labels)

    # create figure
    fig, ax = plt.subplots()

    n_run = len(histories)

    # extract sample numbers
    samples = []
    for history in histories:
        samples.append(np.array(history.get_all_populations()['samples']))

    # create matrix
    n_pop = max(len(sample) for sample in samples)

    matrix = np.zeros((n_pop, n_run))
    for i_sample, sample in enumerate(samples):
        matrix[:len(sample), i_sample] = sample

    # plot bars
    for i_pop in range(n_pop):
        ax.bar(x=np.arange(n_run),
               height=matrix[i_pop, :],
               bottom=np.sum(matrix[:i_pop, :], axis=0))

    # add labels
    ax.set_xticks(np.arange(n_run))
    ax.set_xticklabels(labels, rotation=rotation)
    ax.set_title("Total required samples")
    ax.set_ylabel("Samples")
    ax.set_xlabel("Run")

    fig.tight_layout()

    return ax
