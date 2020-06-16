import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import List, Union
import numpy as np

from ..storage import History
from ..weighted_statistics import weighted_quantile
from ..transition import Transition, MultivariateNormalTransition
from .util import to_lists_or_default


def plot_credible_intervals(
        history: History,
        m: int = 0,
        ts: Union[List[int], int] = None,
        par_names: List = None,
        levels: List = None,
        show_mean: bool = False,
        show_kde_max: bool = False,
        show_kde_max_1d: bool = False,
        size: tuple = None,
        refval: dict = None,
        refval_color: str = 'C1',
        kde: Transition = None,
        kde_1d: Transition = None,
        arr_ax: List = None):
    """
    Plot credible intervals over time.

    Parameters
    ----------

    history: History
        The history to extract data from.
    m: int, optional (default = 0)
        The id of the model to plot for.
    ts: Union[List[int], int], optional (default = all)
        The time points to plot for.
    par_names: List[str], optional
        The parameter to plot for. If None, then all parameters are used.
    levels: List[float], optional (default = [0.95])
        Confidence intervals to compute.
    show_mean: bool, optional (default = False)
        Whether to show the mean apart from the median as well.
    show_kde_max: bool, optional (default = False)
        Whether to show the one of the sampled points that gives the highest
        KDE value for the specified KDE.
        Note: It is not attemtped to find the overall hightest KDE value, but
        rather the sampled point with the highest value is taken as an
        approximation (of the MAP-value).
    show_kde_max_1d: bool, optional (default = False)
        Same as `show_kde_max`, but here the KDE is applied componentwise.
    size: tuple of float
        Size of the plot.
    refval: dict, optional (default = None)
        A dictionary of reference parameter values to plot for each of
        `par_names`.
    refval_color: str, optional
        Color to use for the reference value.
    kde: Transition, optional (default = MultivariateNormalTransition)
        The KDE to use for `show_kde_max`.
    kde_1d: Transition, optional (default = MultivariateNormalTransition)
        The KDE to use for `show_kde_max_1d`.
    arr_ax: List, optional
        Array of axes to use. Assumed to be a 1-dimensional list.

    Returns
    -------

    arr_ax: Array of generated axes.
    """
    if levels is None:
        levels = [0.95]
    levels = sorted(levels)
    if par_names is None:
        # extract all parameter names
        df, _ = history.get_distribution(m=m)
        par_names = list(df.columns.values)
    # dimensions
    n_par = len(par_names)
    n_confidence = len(levels)
    if ts is None:
        ts = list(range(0, history.max_t + 1))
    n_pop = len(ts)

    # prepare axes
    if arr_ax is None:
        _, arr_ax = plt.subplots(
            nrows=n_par, ncols=1, sharex=False, sharey=False, figsize=size)
    if not isinstance(arr_ax, (list, np.ndarray)):
        arr_ax = [arr_ax]
    fig = arr_ax[0].get_figure()

    # prepare matrices
    cis = np.empty((n_par, n_pop, 2 * n_confidence))
    median = np.empty((n_par, n_pop))
    if show_mean:
        mean = np.empty((n_par, n_pop))
    if show_kde_max:
        kde_max = np.empty((n_par, n_pop))
    if show_kde_max_1d:
        kde_max_1d = np.empty((n_par, n_pop))
    if kde is None and show_kde_max:
        kde = MultivariateNormalTransition()
    if kde_1d is None and show_kde_max_1d:
        kde_1d = MultivariateNormalTransition()

    # fill matrices
    # iterate over populations
    for i_t, t in enumerate(ts):
        df, w = history.get_distribution(m=m, t=t)
        # normalize weights to be sure
        w /= w.sum()
        # fit kde
        if show_kde_max:
            _kde_max_pnt = compute_kde_max(kde, df, w)
        # iterate over parameters
        for i_par, par in enumerate(par_names):
            # as numpy array
            vals = np.array(df[par])
            # median
            median[i_par, i_t] = compute_quantile(vals, w, 0.5)
            # mean
            if show_mean:
                mean[i_par, i_t] = np.sum(w * vals)
            # kde max
            if show_kde_max:
                kde_max[i_par, i_t] = _kde_max_pnt[par]
            if show_kde_max_1d:
                _kde_max_1d_pnt = compute_kde_max(kde_1d, df[[par]], w)
                kde_max_1d[i_par, i_t] = _kde_max_1d_pnt[par]
            # levels
            for i_c, confidence in enumerate(levels):
                lb, ub = compute_credible_interval(
                    vals, w, confidence)
                cis[i_par, i_t, i_c] = lb
                cis[i_par, i_t, -1 - i_c] = ub

    # plot
    for i_par, (par, ax) in enumerate(zip(par_names, arr_ax)):
        for i_c, confidence in reversed(list(enumerate(levels))):
            ax.errorbar(
                x=range(n_pop),
                y=median[i_par].flatten(),
                yerr=[median[i_par] - cis[i_par, :, i_c],
                      cis[i_par, :, -1 - i_c] - median[i_par]],
                capsize=(5.0 / n_confidence) * (i_c + 1),
                label="{:.2f}".format(confidence))
        ax.set_title(f"Parameter {par}")
        # mean
        if show_mean:
            ax.plot(range(n_pop), mean[i_par], 'x-', label="Mean")
        # kde max
        if show_kde_max:
            ax.plot(range(n_pop), kde_max[i_par], 'x-', label="Max KDE")
        if show_kde_max_1d:
            ax.plot(range(n_pop), kde_max_1d[i_par], 'x-',
                    label="Max KDE 1d")
        # reference value
        if refval is not None:
            ax.hlines(refval[par], xmin=0, xmax=n_pop - 1, color=refval_color,
                      label="Reference value")
        ax.set_xticks(range(n_pop))
        ax.set_xticklabels(ts)
        ax.set_ylabel(par)
        ax.legend()

    # format
    arr_ax[-1].set_xlabel("Population t")
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return arr_ax


def plot_credible_intervals_for_time(
        histories: Union[List[History], History],
        labels: Union[List[str], str] = None,
        ms: Union[List[int], int] = None,
        ts: Union[List[int], int] = None,
        par_names: List[str] = None,
        levels: List[float] = None,
        show_mean: bool = False,
        show_kde_max: bool = False,
        show_kde_max_1d: bool = False,
        size: tuple = None,
        rotation: int = 0,
        refvals: Union[List[dict], dict] = None,
        kde: Transition = None,
        kde_1d: Transition = None):
    """
    Plot credible intervals over time.

    Parameters
    ----------

    histories: Union[List[History], History]
        The histories to extract data from.
    labels: Union[List[str], str], optional
        Labels for the histories. If None, they are just numbered.
    ms: Union[List[int], int], optional (default = 0)
        List of the ids of the models to plot for. Default is
        model id 0 for all histories.
    ts: Union[List[int], int], optional (default = all)
        The time points to plot for, same length as histories.
        If None, the last times are taken.
    par_names: List[str], optional
        The parameter to plot for. If None, then all parameters are used.
        Assumes all histories have these parameters.
    levels: List[float], optional (default = [0.95])
        Confidence intervals to compute.
    show_mean, show_kde_max, show_kde_max_1d: bool, optional (default = False)
        As in `plot_credible_intervals`.
    size: tuple of float
        Size of the plot.
    refvals: Union[List[dict], dict], optional (default = None)
        A dictionary of reference parameter values to plot for each of
        `par_names`, for each history. Same length as histories.
    kde: Transition, optional (default = MultivariateNormalTransition)
        The KDE to use for `show_kde_max`.
    kde_1d: Transition, optional (default = MultivariateNormalTransition)
        The KDE to use for `show_kde_max_1d`.
    """
    histories, labels = to_lists_or_default(histories, labels)
    n_run = len(histories)
    if ms is None:
        ms = [0] * n_run
    elif not isinstance(ms, list) or len(ms) == 1:
        ms = [ms] * n_run
    if levels is None:
        levels = [0.95]
    levels = sorted(levels)
    if par_names is None:
        # extract all parameter names
        df, _ = histories[0].get_distribution(m=ms[0])
        par_names = list(df.columns.values)
    n_par = len(par_names)
    n_confidence = len(levels)
    if ts is None:
        ts = [history.max_t for history in histories]
    if refvals is not None and not isinstance(refvals, list):
        refvals = [refvals] * n_run

    # prepare axes
    fig, arr_ax = plt.subplots(
        nrows=n_par, ncols=1, sharex=False, sharey=False)
    if n_par == 1:
        arr_ax = [arr_ax]

    # prepare matrices
    cis = np.empty((n_par, n_run, 2 * n_confidence))
    median = np.empty((n_par, n_run))
    if show_mean:
        mean = np.empty((n_par, n_run))
    if show_kde_max:
        kde_max = np.empty((n_par, n_run))
    if show_kde_max_1d:
        kde_max_1d = np.empty((n_par, n_run))
    if kde is None and show_kde_max:
        kde = MultivariateNormalTransition()
    if kde_1d is None and show_kde_max_1d:
        kde_1d = MultivariateNormalTransition()

    # fill matrices
    # iterate over populations
    for i_run, (h, t, m) in enumerate(zip(histories, ts, ms)):
        df, w = h.get_distribution(m=m, t=t)
        # normalize weights to be sure
        w /= w.sum()
        # fit kde
        if show_kde_max:
            _kde_max_pnt = compute_kde_max(kde, df, w)
        # iterate over parameters
        for i_par, par in enumerate(par_names):
            # as numpy array
            vals = np.array(df[par])
            # median
            median[i_par, i_run] = compute_quantile(vals, w, 0.5)
            # mean
            if show_mean:
                mean[i_par, i_run] = np.sum(w * vals)
            # kde max
            if show_kde_max:
                kde_max[i_par, i_run] = _kde_max_pnt[par]
            if show_kde_max_1d:
                _kde_max_1d_pnt = compute_kde_max(kde_1d, df[[par]], w)
                kde_max_1d[i_par, i_run] = _kde_max_1d_pnt[par]
            # levels
            for i_c, confidence in enumerate(levels):
                lb, ub = compute_credible_interval(
                    vals, w, confidence)
                cis[i_par, i_run, i_c] = lb
                cis[i_par, i_run, -1 - i_c] = ub

    # plot
    for i_par, (par, ax) in enumerate(zip(par_names, arr_ax)):
        for i_run in range(len(histories)):
            for i_c in reversed(range(len(levels))):
                y_err = np.array(
                    [median[i_par, i_run] - cis[i_par, i_run, i_c],
                     cis[i_par, i_run, -1 - i_c] - median[i_par, i_run]])
                y_err = y_err.reshape((2, 1))
                ax.errorbar(
                    x=[i_run],
                    y=median[i_par, i_run],
                    yerr=y_err,
                    capsize=(10.0 / n_confidence) * (i_c + 1),
                    color=f'C{i_c}',
                )
            # reference value
            if refvals[i_run] is not None:
                ax.plot([i_run], [refvals[i_run][par]], 'x',
                        color='black')
        ax.set_title(f"Parameter {par}")
        # mean
        if show_mean:
            ax.plot(range(n_run), mean[i_par], 'x',
                    color=f'C{n_confidence}')
        # kde max
        if show_kde_max:
            ax.plot(range(n_run), kde_max[i_par], 'x',
                    color=f'C{n_confidence + 1}')
        if show_kde_max_1d:
            ax.plot(range(n_run), kde_max_1d[i_par], 'x',
                    color=f'C{n_confidence + 2}')
        ax.set_xticks(range(n_run))
        ax.set_xticklabels(labels, rotation=rotation)
        leg_colors = [f'C{i_c}' for i_c in reversed(range(n_confidence))]
        leg_labels = ['{:.2f}'.format(c) for c in reversed(levels)]
        if show_mean:
            leg_colors.append(f'C{n_confidence}')
            leg_labels.append("Mean")
        if show_kde_max:
            leg_colors.append(f'C{n_confidence + 1}')
            leg_labels.append("Max KDE")
        if show_kde_max_1d:
            leg_colors.append(f'C{n_confidence + 2}')
            leg_labels.append("Max KDE 1d")
        if refvals is not None:
            leg_colors.append('black')
            leg_labels.append("Reference value")
        handles = [Line2D([0], [0], color=c, label=l)
                   for c, l in zip(leg_colors, leg_labels)]
        ax.legend(handles=handles, bbox_to_anchor=(1.04, 1), loc="upper left")

    # format
    arr_ax[-1].set_xlabel("Population t")
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return arr_ax


def compute_credible_interval(vals, weights, confidence: float = 0.95):
    """
    Compute credible interval to confidence level `confidence` for points
    `vals` associated to weights `weights`.

    Returns
    -------
    lb, ub: tuple of float
        Lower and upper bound of the credible interval.
    """
    if confidence <= 0.0 or confidence >= 1.0:
        raise ValueError(
            f"Confidence {confidence} must be in the interval (0.0, 1.0).")
    alpha_lb = 0.5 * (1.0 - confidence)
    alpha_ub = confidence + alpha_lb
    lb = compute_quantile(vals, weights, alpha_lb)
    ub = compute_quantile(vals, weights, alpha_ub)
    return lb, ub


def compute_kde_max(kde, df, w):
    """
    Fit the kde and find the maximal kde value among the points in df.
    """
    kde.fit(df, w)
    kde_vals = [kde.pdf(p) for _, p in df.iterrows()]
    ix = kde_vals.index(max(kde_vals))
    kde_max_pnt = df.iloc[ix]
    return kde_max_pnt


def compute_quantile(vals, weights, alpha):
    """
    Compute `alpha`-quantile for points `vals` associated to weights
    `weights`.
    """
    return weighted_quantile(vals, weights, alpha=alpha)
