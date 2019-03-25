"""
To plot densities from the weighted importance samples, the visualization
routines employ a kernel density estimate. Note that this can "over-smoothen"
so that local structure is lost. If this could be the case, it makes sense
to in the argument `kde` reduce the `scaling` in the default
MultivariateNormalTransition(), or to replace it by a GridSearchCV() to
automatically find a visually good level of smoothness.
"""

import numpy as np
from ..transition import MultivariateNormalTransition
import matplotlib.pyplot as plt
import pandas as pd


def kde_1d(df, w, x, xmin=None, xmax=None, numx=50, kde=None):
    """
    Calculates a 1 dimensional histogram from a Dataframe and weights.

    For example, a results distribution might be obtained from the history
    class and plotted as follows::

        df, w = history.get_distribution(0)
        x, pdf = hist_2d(df, w, "x")
        plt.plot(x, pdf)


    Parameters
    ----------
    df: Pandas Dataframe
        The rows are the observations, the columns the variables
    w: np.ndarray
        The corresponding weights
    x: str
        The variable for the x-axis
    xmin: float, optional
        The lower limit in x for the histogram.
        If left empty, it is set to the minimum of the ovbservations of the
        variable to be plotted as x.
    xmax: float, optional
        The upper limit in x for the histogram.
        If left empty, it is set to the maximum of the ovbservations of the
        variable to be plotted as x.
    numx: int, optional
        The number of bins in x direction.
        Defaults to 50.
    kde: pyabc.Transition, optional
        The kernel density estimator to use for creating a smooth density
        from the sample. If None, a multivariate normal kde with
        cross-validated scaling is used.

    Returns
    -------

    x, pdf: (np.ndarray, np.ndarray)
        The x and the densities at these points.
        These can be passed for plotting, for example as
        plt.plot(x, pdf)

    """
    if kde is None:
        kde = MultivariateNormalTransition(scaling=1)
    kde.fit(df[[x]], w)

    if xmin is None:
        xmin = df[x].min()
    if xmax is None:
        xmax = df[x].max()
    x_vals = np.linspace(xmin, xmax, num=numx)
    test = pd.DataFrame({x: x_vals})
    pdf = kde.pdf(test)
    return x_vals, pdf


def plot_kde_1d(df, w, x, xmin=None, xmax=None,
                numx=50, ax=None, title: str = None,
                refval=None, kde=None, **kwargs):
    """
    Plots a 1d histogram.

    Parameters
    ----------
    df: Pandas Dataframe
        The rows are the observations, the columns the variables
    w: The corresponding weights
    x: str
        The variable for the x-axis

    xmin: float, optional
        The lower limit in x for the histogram.
        If left empty, it is set to the minimum of the ovbservations of the
        variable to be plotted as x.
    xmax: float, optional
        The upper limit in x for the histogram.
        If left empty, it is set to the maximum of the ovbservations of the
        variable to be plotted as x.
    numx: int, optional
        The number of bins in x direction.
        Defaults tp 50.
    title: str, optional
        Title for the plot. Defaults to None.
    refval: dict, optional
        A reference value for x (as refval[x]: float).
        If not None, the value will be highlighted in the plot.
        Default: None.
    kde: pyabc.Transition, optional
        The kernel density estimator to use for creating a smooth density
        from the sample. If None, a multivariate normal kde with
        cross-validated scaling is used.

    Returns
    -------

    ax: matplotlib axis
        axis of the plot

    """
    x_vals, pdf = kde_1d(df, w, x, xmin=xmin, xmax=xmax,  numx=numx, kde=kde)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x_vals, pdf, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel("Posterior")
    ax.set_xlim(xmin, xmax)
    if title is not None:
        ax.set_title(title)
    if refval is not None:
        ax.axvline(refval[x], color='C1', linestyle='dashed')
    return ax


def kde_2d(df, w, x, y, xmin=None, xmax=None, ymin=None, ymax=None,
           numx=50, numy=50, kde=None):
    """
    Calculates a 2 dimensional histogram from a Dataframe and weights.

    For example, a results distribution might be obtained from the history
    class and plotted as follows::

        df, w = history.get_distribution(0)
        X, Y, PDF = hist_2d(df, w, "x", "y")
        plt.pcolormesh(X, Y, PDF)


    Parameters
    ----------
    df: Pandas Dataframe
        The rows are the observations, the columns the variables
    w: The corresponding weights
    x: str
        The variable for the x-axis
    y: str
        The variable for the y-axis
    xmin: float, optional
        The lower limit in x for the histogram.
        If left empty, it is set to the minimum of the ovbservations of the
        variable to be plotted as x.
    xmax: float, optional
        The upper limit in x for the histogram.
        If left empty, it is set to the maximum of the ovbservations of the
        variable to be plotted as x.
    ymin: float, optional
        The lower limit in y for the histogram.
        If left empty, it is set to the minimum of the ovbservations of the
        variable to be plotted as y
    ymax: float, optional
        The upper limit in y for the histogram.
        If left empty, it is set to the maximum of the ovbservations of the
        variable to be plotted as y.
    numx: int, optional
        The number of bins in x direction.
        Defaults tp 50.
    numy int, optional
        The number of bins in y direction.
        Defaults tp 50.
    kde: pyabc.Transition, optional
        The kernel density estimator to use for creating a smooth density
        from the sample. If None, a multivariate normal kde with
        cross-validated scaling is used.

    Returns
    -------

    X, Y, PDF: (np.ndarray, np.ndarray, np.ndarray)
        The X, the Y and the densities at these points.
        These can be passed for plotting, for example as
        plt.pcolormesh(X, Y, PDF)

    """
    if kde is None:
        kde = MultivariateNormalTransition(scaling=1)
    kde.fit(df[[x, y]], w)

    if xmin is None:
        xmin = df[x].min()
    if xmax is None:
        xmax = df[x].max()
    if ymin is None:
        ymin = df[y].min()
    if ymax is None:
        ymax = df[y].max()
    X, Y = np.meshgrid(np.linspace(xmin, xmax, num=numx),
                       np.linspace(ymin, ymax, num=numy))
    test = pd.DataFrame({x: X.flatten(), y: Y.flatten()})
    pdf = kde.pdf(test)
    PDF = pdf.reshape(X.shape)
    return X, Y, PDF


def plot_kde_2d(df, w, x, y, xmin=None, xmax=None, ymin=None, ymax=None,
                numx=50, numy=50, ax=None, colorbar=True,
                title: str = None, refval=None, kde=None, **kwargs):
    """
    Plots a 2d histogram.

    Parameters
    ----------
    df: Pandas Dataframe
        The rows are the observations, the columns the variables
    w: The corresponding weights.
    x: str
        The variable for the x-axis.
    y: str
        The variable for the y-axis.
    xmin: float, optional
        The lower limit in x for the histogram.
        If left empty, it is set to the minimum of the ovbservations of the
        variable to be plotted as x.
    xmax: float, optional
        The upper limit in x for the histogram.
        If left empty, it is set to the maximum of the ovbservations of the
        variable to be plotted as x.
    ymin: float, optional
        The lower limit in y for the histogram.
        If left empty, it is set to the minimum of the ovbservations of the
        variable to be plotted as y.
    ymax: float, optional
        The upper limit in y for the histogram.
        If left empty, it is set to the maximum of the ovbservations of the
        variable to be plotted as y.
    numx: int, optional
        The number of bins in x direction.
        Defaults to 50.
    numy int, optional
        The number of bins in y direction.
        Defaults tp 50.
    colorbar: bool, optional
        Whether to plot a colorbar. Defaults to True.
    title: str, optional
        Title for the plot. Defaults to None.
    refval: dict, optional
        A reference parameter to be shown in the plots. Default: None.
    kde: pyabc.Transition, optional
        The kernel density estimator to use for creating a smooth density
        from the sample. If None, a multivariate normal kde with
        cross-validated scaling is used.

    Returns
    -------

    ax: matplotlib axis
        axis of the plot

    """
    X, Y, PDF = kde_2d(df, w, x, y,
                       xmin=xmin, xmax=xmax,
                       ymin=ymin, ymax=ymax, numx=numx, numy=numy,
                       kde=kde)
    if ax is None:
        _, ax = plt.subplots()
    mesh = ax.pcolormesh(X, Y, PDF, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title is not None:
        ax.set_title(title)
    if colorbar:
        plt.colorbar(mesh, ax=ax)
        # cbar.set_label("PDF")
    if refval is not None:
        ax.scatter([refval[x]], [refval[y]], color='C1')
    return ax


def plot_kde_matrix(df, w,
                    limits=None,
                    colorbar=True,
                    height=2.5,
                    numx=50,
                    numy=50,
                    refval=None,
                    kde=None):
    """
    Plot a KDE matrix.

    Parameters
    ----------

    df: Pandas Dataframe
        The rows are the observations, the columns the variables.
    w: np.narray
        The corresponding weights.
    colorbar: bool
        Whether to plot the colorbars or not.
    limits: dictionary, optional
        Dictionary of the form ``{"name": (lower_limit, upper_limit)}``.
    height: float, optional
        Height of each subplot in inches. Default: 2.5.
    numx: int, optional
        The number of bins in x direction.
        Defaults to 50.
    numy: int, optional
        The number of bins in y direction.
        Defaults to 50.
    refval: dict, optional
        A reference parameter to be shown in the plots (e.g. the
        underlying ground truth parameter used to simulate the data
        for testing purposes). Default: None.
    kde: pyabc.Transition, optional
        The kernel density estimator to use for creating a smooth density
        from the sample. If None, a multivariate normal kde with
        cross-validated scaling is used.
    """

    n_par = df.shape[1]
    par_names = list(df.columns.values)
    fig, arr_ax = plt.subplots(nrows=n_par, ncols=n_par,
                               sharex=False, sharey=False,
                               figsize=(height * n_par, height * n_par))

    if limits is None:
        limits = {}
    default = (None, None)

    def hist_2d(x, y, ax):
        df = pd.concat((x, y), axis=1)
        plot_kde_2d(df, w,
                    x.name, y.name,
                    xmin=limits.get(x.name, default)[0],
                    xmax=limits.get(x.name, default)[1],
                    ymin=limits.get(y.name, default)[0],
                    ymax=limits.get(y.name, default)[1],
                    numx=numx, numy=numy,
                    ax=ax, title=False, colorbar=colorbar,
                    refval=refval, kde=kde)

    def scatter(x, y, ax):
        alpha = w / w.max()
        colors = np.zeros((alpha.size, 4))
        colors[:, 3] = alpha
        ax.scatter(x, y, color="k")
        if refval is not None:
            ax.scatter([refval[x.name]], [refval[y.name]], color='C1')
        ax.set_xlim(*limits.get(x.name, default))
        ax.set_ylim(*limits.get(y.name, default))

    def hist_1d(x, ax):
        df = pd.concat((x,), axis=1)
        plot_kde_1d(df, w, x.name,
                    xmin=limits.get(x.name, default)[0],
                    xmax=limits.get(x.name, default)[1],
                    numx=numx,
                    ax=ax, refval=refval, kde=kde)

    # fill all subplots
    for i in range(0, n_par):
        y_name = par_names[i]
        y = df[y_name]

        # diagonal
        ax = arr_ax[i, i]
        hist_1d(y, ax)

        for j in range(0, i):
            x_name = par_names[j]
            x = df[x_name]

            # lower
            ax = arr_ax[i, j]
            hist_2d(x, y, ax)

            # upper
            ax = arr_ax[j, i]
            scatter(y, x, ax)

    # format
    _format_kde_matrix(arr_ax, par_names)

    # adjust subplots to fit
    fig.tight_layout()

    return arr_ax


def _format_kde_matrix(arr_ax, par_names):
    """
    Clear all labels and legends, and set the left-most and bottom-most
    labels to the parameter names.

    Parameters
    ----------

    arr_ax: array of matplotlib.axes.Axes
        Shape (n_par, n_par) where len(par_names) == n_par.
    par_names: list of str
        Parameter names to be used as labels.
    """
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
