"""Sample number plots"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.axes
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from typing import List, Union

from ..storage import History
from .util import to_lists, get_labels
from ..weighted_statistics import effective_sample_size


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
    histories = to_lists(histories)
    labels = get_labels(labels, len(histories))

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
    for i_pop in reversed(range(n_pop)):
        ax.bar(x=np.arange(n_run),
               height=matrix[i_pop, :],
               bottom=np.sum(matrix[:i_pop, :], axis=0),
               label=f"Generation {i_pop-1}")

    # add labels
    ax.set_xticks(np.arange(n_run))
    ax.set_xticklabels(labels, rotation=rotation)
    ax.set_title(title)
    ax.set_ylabel("Samples")
    ax.set_xlabel("Run")
    ax.legend()
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
    histories = to_lists(histories)
    labels = get_labels(labels, len(histories))

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
    histories = to_lists(histories)
    labels = get_labels(labels, len(histories))

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
    if any(lab is not None for lab in labels):
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
        title: str = "Acceptance rates",
        yscale: str = 'lin',
        size: tuple = None,
        ax: mpl.axes.Axes = None,
        colors: List[str] = None,
        normalize_by_ess: bool = False):
    """
    Plot of acceptance rates over all iterations, i.e. one trajectory
    per history.

    Parameters
    ----------
    histories:
        The histories to plot from. History ids must be set correctly.
    labels:
        Labels corresponding to the histories. If None are provided,
        indices are used as labels.
    title:
        Title for the plot.
    yscale:
        The scale on which to plot the counts. Can be one of 'lin', 'log'
        (basis e) or 'log10'
    size:
        The size of the plot in inches.
    ax:
        The axis object to use.
    normalize_by_ess: bool, optional (default = False)
        Indicator to use effective sample size for the acceptance rate in
        place of the population size.

    Returns
    -------
    ax: Axis of the generated plot.
    """
    # preprocess input
    histories = to_lists(histories)
    labels = get_labels(labels, len(histories))
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
        if normalize_by_ess:
            ess = np.zeros(len(h_info['t']) - 1)
            for t in np.array(h_info['t'])[1:]:
                w = history.get_weighted_distances(t=t)['w']
                ess[t-1] = effective_sample_size(w)
            pop_sizes.append(ess)
        else:
            pop_sizes.append(np.array(
                history.get_nr_particles_per_population().values[1:]))
        samples.append(np.array(h_info['samples'])[1:])

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


def plot_lookahead_evaluations(
        sampler_df: Union[pd.DataFrame, str],
        relative: bool = False,
        fill: bool = False,
        alpha: float = None,
        t_min: int = 0,
        title: str = "Total evaluations",
        size: tuple = None,
        ax: mpl.axes.Axes = None):
    """Plot total vs look-ahead evaluations over the generations.

    Parameters
    ----------
    sampler_df:
        Dataframe or file as generated via
        `RedisEvalParallelSampler(log_file=...)`.
    relative:
        Whether to normalize the total evaluations for each generation to 1.
    fill:
        If True, instead of lines, filled areas are drawn that sum up to the
        totals.
    alpha:
        Alpha value for lines or areas.
    t_min:
        The minimum generation to show. E.g. a value of 1 omits the first
        generation.
    title:
        Plot title.
    size:
        The size of the plot in inches.
    ax:
        The axis object to use.

    Returns
    -------
    ax: Axis of the generated plot.
    """
    # process input
    if isinstance(sampler_df, str):
        sampler_df = pd.read_csv(sampler_df, sep=',')
    if alpha is None:
        alpha = 0.7 if fill else 1.0

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # restrict to t >= 0
    sampler_df = sampler_df[sampler_df.t >= t_min]

    # extract variables
    t = sampler_df.t
    n_la = sampler_df.n_lookahead
    n_eval = sampler_df.n_evaluated
    n_act = n_eval - n_la

    # normalize
    if relative:
        n_la /= n_eval
        n_act /= n_eval
        n_eval /= n_eval

    # plot
    if fill:
        ax.fill_between(t, n_la, n_eval, alpha=alpha, label="Actual")
        ax.fill_between(t, 0, n_la, alpha=alpha, label="Look-ahead")
    else:
        ax.plot(t, n_eval, linestyle='--', marker='o', color='black',
                alpha=alpha, label="Total")
        ax.plot(t, n_act, marker='o', alpha=alpha, label="Actual")
        ax.plot(t, n_la, marker='o', alpha=alpha, label="Look-ahead")

    # prettify plot
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Population index")
    ax.set_ylabel("Evaluations")
    ax.set_ylim(bottom=0)
    # enforce integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if size is not None:
        fig.set_size_inches(size)

    return ax


def plot_lookahead_final_acceptance_fractions(
        sampler_df: Union[pd.DataFrame, str],
        population_sizes: Union[np.ndarray, History],
        relative: bool = False,
        fill: bool = False,
        alpha: float = None,
        t_min: int = 0,
        title: str = "Composition of final acceptances",
        size: tuple = None,
        ax: mpl.axes.Axes = None):
    """Plot fraction of look-ahead samples in final acceptances,
    over generations.

    Parameters
    ----------
    sampler_df:
        Dataframe or file as generated via
        `RedisEvalParallelSampler(log_file=...)`.
    population_sizes:
        The sizes of the populations of accepted particles. If a History is
        passed, those values are extracted automatically, otherwise should
        be for the same time values as `sampler_df`.
    relative:
        Whether to normalize the total evaluations for each generation to 1.
    fill:
        If True, instead of lines, filled areas are drawn that sum up to the
        totals.
    alpha:
        Alpha value for lines or areas.
    t_min:
        The minimum generation to show. E.g. a value of 1 omits the first
        generation.
    title:
        Plot title.
    size:
        The size of the plot in inches.
    ax:
        The axis object to use.

    Returns
    -------
    ax: Axis of the generated plot.
    """
    # process input
    if isinstance(sampler_df, str):
        sampler_df = pd.read_csv(sampler_df, sep=',')
    if alpha is None:
        alpha = 0.7 if fill else 1.0

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # get numbers of final acceptances
    if isinstance(population_sizes, History):
        pop = population_sizes.get_all_populations()

        population_sizes = np.array(
            [pop.loc[pop.t == t, 'particles'] for t in sampler_df.t],
            dtype=float).flatten()

    # restrict to t >= 0
    population_sizes = population_sizes[sampler_df.t >= t_min]
    sampler_df = sampler_df[sampler_df.t >= t_min]

    # extract variables
    t = sampler_df.t

    n_la_acc = sampler_df.n_lookahead_accepted
    # actual look-ahead acceptances cannot be more than requested
    n_la_acc = np.minimum(n_la_acc, population_sizes)

    # actual acceptances are the remaining ones, as these are always later
    n_act_acc = population_sizes - n_la_acc

    # normalize
    if relative:
        n_la_acc /= population_sizes
        n_act_acc /= population_sizes
        population_sizes /= population_sizes

    # plot
    if fill:
        ax.fill_between(t, n_la_acc, population_sizes, alpha=alpha,
                        label="Actual")
        ax.fill_between(t, 0, n_la_acc, alpha=alpha, label="Look-ahead")
    else:
        ax.plot(t, population_sizes, linestyle='--', marker='o', color='black',
                alpha=alpha, label="Population size")
        ax.plot(t, n_act_acc, marker='o', alpha=alpha, label="Actual")
        ax.plot(t, n_la_acc, marker='o', alpha=alpha, label="Look-ahead")

    # prettify plot
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Population index")
    ax.set_ylabel("Final acceptances")
    ax.set_ylim(bottom=0)
    # enforce integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if size is not None:
        fig.set_size_inches(size)

    return ax


def plot_lookahead_acceptance_rates(
        sampler_df: Union[pd.DataFrame, str],
        t_min: int = 0,
        title: str = "Acceptance rates",
        size: tuple = None,
        ax: mpl.axes.Axes = None):
    """Plot acceptance rates for look-ahead vs ordinary samples.
    The ratios are relative to all accepted particles, including eventually
    discarded ones.

    Parameters
    ----------
    sampler_df:
        Dataframe or file as generated via
        `RedisEvalParallelSampler(log_file=...)`.
    t_min:
        The minimum generation to show. E.g. a value of 1 omits the first
        generation.
    title:
        Plot title.
    size:
        The size of the plot in inches.
    ax:
        The axis object to use.

    Returns
    -------
    ax: Axis of the generated plot.
    """
    # process input
    if isinstance(sampler_df, str):
        sampler_df = pd.read_csv(sampler_df, sep=',')

    # create figure
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    # restrict to t >= 0
    sampler_df = sampler_df[sampler_df.t >= t_min]

    # extract variables

    # time
    t = sampler_df.t

    # look-ahead acceptances and samples
    n_la_acc = sampler_df.n_lookahead_accepted
    n_la = sampler_df.n_lookahead

    # total acceptances and samples
    n_all_acc = sampler_df.n_accepted
    n_all = sampler_df.n_evaluated

    # difference (actual proposal)
    n_act_acc = n_all_acc - n_la_acc
    n_act = n_all - n_la

    # plot
    ax.plot(t, n_all_acc / n_all, linestyle='--', marker='o', color='black',
            label="Combined")
    ax.plot(t, n_act_acc / n_act, marker='o', label="Actual")
    ax.plot(t, n_la_acc / n_la, marker='o', label="Look-ahead")

    # prettify plot
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Population index")
    ax.set_ylabel("Acceptance rate")
    ax.set_ylim(bottom=0)
    # enforce integer ticks
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if size is not None:
        fig.set_size_inches(size)

    return ax
