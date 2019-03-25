import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List
import numpy as np

from ..storage import History
from ..weighted_statistics import weighted_quantile


def plot_confidence_intervals(
        history: History,
        m: int = 0,
        par_names: List = None,
        confidences: List = None,
        show_mean: bool = False,
        size: tuple = None,):
    """
    Plot confidence intervals over time.

    Parameters
    ----------

    history: History
        The history to extract data from.
    m: int, optional (default = 0)
        The id of the model to plot for.
    par_names: List of str, optional
        The parameter to plot for. If None, then all parameters are used.
    show_mean: bool, optional (default = False)
        Whether to show the mean apart from the median as well.
    size: tuple of float
        Size of the plot.
    """
    if confidences is None:
        confidences = [0.95]
    confidences = sorted(confidences)
    if par_names is None:
        # extract all parameter names
        df, _ = history.get_distribution(m=m)
        par_names = list(df.columns.values)

    # dimensions
    n_par = len(par_names)
    n_pop = history.max_t + 1
    n_confidence = len(confidences)

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

    # fill matrices
    # iterate over populations
    for t in range(0, n_pop):
        df, w = history.get_distribution(m=m, t=t)
        # normalize weights to be sure
        w /= w.sum()
        # iterate over parameters
        for i_par, par in enumerate(par_names):
            vals = np.array(df[par])
            # median
            median[i_par, t] = compute_quantile(vals, w, 0.5)
            # mean
            if show_mean:
                mean[i_par, t] = np.sum(w * vals)
            # confidences
            for i_c, confidence in enumerate(confidences):
                lb, ub = compute_confidence_interval(
                    vals, w, confidence)
                cis[i_par, t, i_c] = lb
                cis[i_par, t, -1 - i_c] = ub

    # plot
    for i_par, (par, ax) in enumerate(zip(par_names, arr_ax)):
        for i_c, confidence in reversed(list(enumerate(confidences))):
            ax.errorbar(
                x=range(0, n_pop),
                y=median[i_par].flatten(),
                yerr=[median[i_par] - cis[i_par, :, i_c],
                      cis[i_par, :, -1 - i_c] - median[i_par]],
                capsize=(5.0 / n_confidence) * (i_c + 1),
                label="{:.2f}".format(confidence))
        ax.set_title(f"Parameter {par}")
        ax.legend()

    # format
    arr_ax[-1].set_xlabel("Population t")
    for ax in arr_ax:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()


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


def compute_quantile(vals, weights, alpha):
    """
    Compute `alpha`-quantile for points `vals` associated to weights
    `weights`.
    """
    return weighted_quantile(vals, weights, alpha=alpha)
