import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from typing import List, Union

from ..storage import History
from .util import to_lists_or_default


def plot_sample_numbers(
        histories: Union[List, History],
        labels: Union[List, str] = None,
        rotation: int = 0,
        title: str = "Required samples",
        size: tuple = None,
        ax: mpl.axes.Axes = None):
    """
    Stacked bar plot of required numbers of samples over all iterations.

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
    ax: matplotlib.axes.Axes, optional
        The axis object to use.

    Returns
    -------

    ax: Axis of the generated plot.
    """
    # preprocess input
    histories, labels = to_lists_or_default(histories, labels)

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    n_run = len(histories)

    # extract sample numbers
    samples = []
    for history in histories:
        # note: the first entry corresponds to the calibration and should
        # be included here to be fair against methods not requiring
        # calibration
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
    ax.set_title(title)
    ax.set_ylabel("Samples")
    ax.set_xlabel("Run")
    # set size
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return ax


def plot_total_sample_numbers(
        histories: Union[List, History],
        labels: Union[List, str] = None,
        rotation: int = 0,
        title: str = "Total required samples",
        yscale: str = 'lin',
        size: tuple = None,
        ax: mpl.axes.Axes = None):
    """
    Bar plot of total required sample number over all iterations, i.e.
    a single-colored bar per history, in contrast to `plot_sample_numbers`,
    which visually distinguishes iterations.

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
    yscale: str, optional (default = 'lin')
        The scale on which to plot the counts. Can be one of 'lin', 'log'
        (basis e) or 'log10'
    size: tuple of float, optional
        The size of the plot in inches.
    ax: matplotlib.axes.Axes, optional
        The axis object to use.

    Returns
    -------

    ax: Axis of the generated plot.
    """
    # preprocess input
    histories, labels = to_lists_or_default(histories, labels)

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    n_run = len(histories)

    # extract sample numbers
    samples = []
    for history in histories:
        # note: the first entry corresponds to the calibration and should
        # be included here to be fair against methods not requiring
        # calibration
        samples.append(np.sum(history.get_all_populations()['samples']))
    samples = np.array(samples)

    # apply scale
    ylabel = "Total samples"
    if yscale == 'log':
        samples = np.log(samples)
        ylabel = "log(" + ylabel + ")"
    elif yscale == 'log10':
        samples = np.log10(samples)
        ylabel = "log10(" + ylabel + ")"

    # plot bars
    ax.bar(x=np.arange(n_run),
           height=samples)

    # add labels
    ax.set_xticks(np.arange(n_run))
    ax.set_xticklabels(labels, rotation=rotation)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Run")
    # set size
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return ax


def plot_sample_numbers_trajectory(
        histories: Union[List, History],
        labels: Union[List, str] = None,
        rotation: int = 0,
        title: str = "Required samples",
        yscale: str = 'lin',
        size: tuple = None,
        ax: mpl.axes.Axes = None):
    """
    Plot of required sample number over all iterations, i.e. one trajectory
    per history.

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
    title: str, optional (default = "Required samples")
        Title for the plot.
    yscale: str, optional (default = 'lin')
        The scale on which to plot the counts. Can be one of 'lin', 'log'
        (basis e) or 'log10'
    size: tuple of float, optional
        The size of the plot in inches.
    ax: matplotlib.axes.Axes, optional
        The axis object to use.

    Returns
    -------

    ax: Axis of the generated plot.
    """
    # preprocess input
    histories, labels = to_lists_or_default(histories, labels)

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # extract sample numbers
    times = []
    samples = []
    for history in histories:
        # note: the first entry corresponds to the calibration and should
        # be included here to be fair against methods not requiring
        # calibration
        h_info = history.get_all_populations()
        times.append(np.array(h_info['t']))
        samples.append(np.array(h_info['samples']))

    # apply scale
    ylabel = "Samples"
    if yscale == 'log':
        samples = [np.log(sample) for sample in samples]
        ylabel = "log(" + ylabel + ")"
    elif yscale == 'log10':
        samples = [np.log10(sample) for sample in samples]
        ylabel = "log10(" + ylabel + ")"

    # plot
    for t, sample, label in zip(times, samples, labels):
        ax.plot(t, sample, 'x-', label=label)

    # add labels
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Population index $t$")
    # set size
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return ax


def plot_acceptance_rates_trajectory(
        histories: Union[List, History],
        labels: Union[List, str] = None,
        rotation: int = 0,
        title: str = "Acceptance rates",
        yscale: str = 'lin',
        size: tuple = None,
        ax: mpl.axes.Axes = None,
        colors: List[str] = None):
    """
    Plot of acceptance rates over all iterations, i.e. one trajectory
    per history.

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
    title: str, optional (default = "Acceptance rates")
        Title for the plot.
    yscale: str, optional (default = 'lin')
        The scale on which to plot the counts. Can be one of 'lin', 'log'
        (basis e) or 'log10'
    size: tuple of float, optional
        The size of the plot in inches.
    ax: matplotlib.axes.Axes, optional
        The axis object to use.

    Returns
    -------

    ax: Axis of the generated plot.
    """
    # preprocess input
    histories, labels = to_lists_or_default(histories, labels)
    if colors is None:
        colors = [None] * len(histories)
    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # extract sample numbers
    times = []
    samples = []
    pop_sizes = []
    for history in histories:
        # note: the first entry of time -1 is trivial and is thus ignored here
        h_info = history.get_all_populations()
        times.append(np.array(h_info['t'])[1:])
        samples.append(np.array(h_info['samples'])[1:])
        pop_sizes.append(np.array(
            history.get_nr_particles_per_population().values[1:]))

    # compute acceptance rates
    rates = []
    for sample, pop_size in zip(samples, pop_sizes):
        rates.append(pop_size / sample)

    # apply scale
    ylabel = "Acceptance rate"
    if yscale == 'log':
        rates = [np.log(rate) for rate in rates]
        ylabel = "log(" + ylabel + ")"
    elif yscale == 'log10':
        rates = [np.log10(rate) for rate in rates]
        ylabel = "log10(" + ylabel + ")"

    # plot
    for t, rate, label, color in zip(times, rates, labels, colors):
        ax.plot(t, rate, 'x-', label=label, color=color)

    # add labels
    ax.legend()
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Population index $t$")
    # set size
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return ax
