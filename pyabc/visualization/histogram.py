import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_1d(
        history, x, m=0, t=None, xmin=None, xmax=None, ax=None, **kwargs):

    df, w = history.get_distribution(m=m, t=t)

    return plot_histogram_1d_lowlevel(
        df, w, x, xmin, xmax, ax=ax, **kwargs)


def plot_histogram_1d_lowlevel(
        df, w, x, xmin=None, xmax=None, ax=None, **kwargs):

    if ax is None:
        _, ax = plt.subplots()

    if xmin is not None and xmax is not None:
        range_ = (xmin, xmax)
    else:
        range_ = None

    # plot
    ax.hist(x=df[x], range=range_, weights=w, **kwargs)
    ax.set_xlabel(x)

    return ax


def plot_histogram_2d(
        history, x, y, m=0, t=None, xmin=None, xmax=None, ymin=None, ymax=None, ax=None, **kwargs):
    df, w = history.get_distribution(m=m, t=t)

    return plot_histogram_2d_lowlevel(
        df, w, x, y, xmin, xmax, ymin, ymax, ax=ax, **kwargs)


def plot_histogram_2d_lowlevel(
        df, w, x, y, xmin=None, xmax=None, ymin=None, ymax=None, ax=None, **kwargs):

    if ax is None:
        _, ax = plt.subplots()

    xrange_ = yrange_ = None
    if xmin is not None and xmax is not None:
        xrange_ = [xmin, xmax]
    if ymin is not None and ymax is not None:
        yrange_ = [ymin, ymax]
    if xrange_ and yrange_:
        range_ = [xrange_, yrange_]
    else:
        range_ = None

    # plot
    ax.hist2d(x=df[x], y=df[y], range=range_, weights=w, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    return ax


def plot_histogram_matrix(
        history, m=0, t=None, **kwargs):
    df, w = history.get_distribution(m=m, t=t)
    n_par = df.shape[1]
    par_names = list(df.columns.values)

    fig, arr_ax = plt.subplots(nrows=n_par, ncols=n_par, sharex=False, sharey=False)

    def scatter(x, y, ax):
        ax.scatter(x, y, color="k")

    # fill all subplots
    for i in range(0, n_par):
        y_name = par_names[i]
        y = df[y_name]

        # diagonal
        ax = arr_ax[i, i]
        plot_histogram_1d_lowlevel(df, w, y_name, ax=ax, **kwargs)

        for j in range(0, i):
            x_name = par_names[j]
            x = df[x_name]

            # lower
            ax = arr_ax[i, j]
            plot_histogram_2d_lowlevel(df, w, x_name, y_name, ax=ax, **kwargs)

            # upper
            ax = arr_ax[j, i]
            scatter(y, x, ax)

    # format
    _format_histogram_matrix(arr_ax, par_names)
    fig.tight_layout()

    return arr_ax


def _format_histogram_matrix(arr_ax, par_names):
    n_par = len(par_names)

    for i in range(0, n_par):
        for j in range(0, n_par):
            # clear labels
            arr_ax[i, j].set_xlabel("")
            arr_ax[i, j].set_ylabel("")

            # clear legends
            arr_ax[i, j].legend = None

            # remove spines
            arr_ax[i, j].spines['right'].set_visible(False)
            arr_ax[i, j].spines['top'].set_visible(False)

    # set left-most and bottom-most labels to parameter names
    for ax, label in zip(arr_ax[-1, :], par_names):
        ax.set_xlabel(label)
    for ax, label in zip(arr_ax[:, 0], par_names):
        ax.set_ylabel(label)
