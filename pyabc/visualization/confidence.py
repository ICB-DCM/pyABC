import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List, Union
import numpy as np

from ..storage import History
from ..weighted_statistics import weighted_quantile
from ..transition import Transition, MultivariateNormalTransition


def plot_confidence_intervals(
        history: History,
        m: int = 0,
        ts: Union[List[int], int] = None,
        par_names: List = None,
        confidences: List = None,
        show_mean: bool = False,
        show_kde_max: bool = False,
        show_kde_max_1d: bool = False,
        size: tuple = None,
        refval: dict = None,
        kde: Transition = None,
        kde_1d: Transition = None):
    """
    Plot confidence intervals over time.

    Parameters
    ----------

    history: History
        The history to extract data from.
    m: int, optional (default = 0)
        The id of the model to plot for.
    ts: Union[List[int], int], optional (default = all)
        The time points to plot for.
    par_names: List of str, optional
        The parameter to plot for. If None, then all parameters are used.
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
    kde: Transition, optional (default = MultivariateNormalTransition)
        The KDE to use for `show_kde_max`.
    kde_1d: Transition, optional (default = MultivariateNormalTransition)
        The KDE to use for `show_kde_max_1d`.
    """
    if confidences is None:
        confidences = [0.95]
    confidences = sorted(confidences)
    if par_names is None:
        # extract all parameter names
        df, _ = history.get_distribution(m=m)
        par_names = list(df.columns.values)
    n_par = len(par_names)
    n_confidence = len(confidences)
    if ts is None:
        ts = list(range(0, history.max_t + 1))
    n_pop = len(ts)

    # prepare axes
    fig, arr_ax = plt.subplots(
        nrows=n_par, ncols=1, sharex=False, sharey=False)
    if n_par == 1:
        arr_ax = [arr_ax]

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
            # confidences
            for i_c, confidence in enumerate(confidences):
                lb, ub = compute_confidence_interval(
                    vals, w, confidence)
                cis[i_par, i_t, i_c] = lb
                cis[i_par, i_t, -1 - i_c] = ub

    # plot
    for i_par, (par, ax) in enumerate(zip(par_names, arr_ax)):
        for i_c, confidence in reversed(list(enumerate(confidences))):
            ax.errorbar(
                x=ts,
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
            ax.hlines(refval[par], xmin=min(ts), xmax=max(ts),
                      label="Reference value")
        ax.set_xticks(ts)
        ax.legend()

    # format
    arr_ax[-1].set_xlabel("Population t")
    for ax in arr_ax:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if size is not None:
        fig.set_size_inches(size)
    fig.tight_layout()

    return arr_ax


def compute_confidence_interval(vals, weights, confidence: float = 0.95):
    """
    Compute confidence interval to confidence level `confidence` for points
    `vals` associated to weights `weights`.

    Returns
    -------
    lb, ub: tuple of float
        Lower and upper bound of the confidence interval.
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
