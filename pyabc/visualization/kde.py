"""
To plot densities from the weighted importance samples, the visualization
routines employ a kernel density estimate. Note that this can "over-smoothen"
so that local structure is lost. If this could be the case, it makes sense
to in the argument `kde` reduce the `scaling` in the default
MultivariateNormalTransition(), or to replace it by a GridSearchCV() to
automatically find a visually good level of smoothness.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from ..transition import MultivariateNormalTransition
from ..storage import History
from .util import format_plot_matrix


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


def plot_kde_1d_highlevel(
        history: History, x: str, m: int = 0, t: int = None,
        xmin=None, xmax=None, numx=50, ax=None,
        size=None, title: str = None, refval=None, refval_color='C1',
        kde=None, **kwargs):
    """
    Plot 1d kernel density estimate of parameter samples.

    Parameters
    ----------
    history: History
        History to extract data from.
    x: str
        The variable for the x-axis.
    m: int, optional
        Id of the model to plot for.
    t: int, optional
        Time point to plot for. Defaults to last time point.
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
    ax: matplotlib.axes.Axes, optional
        The axis object to use.
    size: 2-Tuple of float, optional
        Size of the plot in inches.
    title: str, optional
        Title for the plot. Defaults to None.
    refval: dict, optional
        A reference value for x (as refval[x]: float).
        If not None, the value will be highlighted in the plot.
        Default: None.
    refval_color: str, optional
        Color to use for the reference value.
    kde: pyabc.Transition, optional
        The kernel density estimator to use for creating a smooth density
        from the sample. If None, a multivariate normal kde with
        cross-validated scaling is used.

    Returns
    -------
    ax: matplotlib axis
        axis of the plot
    """
    df, w = history.get_distribution(m=m, t=t)

    return plot_kde_1d(df, w, x, xmin, xmax, numx, ax, size, title, refval,
                       refval_color, kde, **kwargs)


def plot_kde_1d(df, w, x, xmin=None, xmax=None,
                numx=50, ax=None, size=None, title: str = None,
                refval=None, refval_color='C1', kde=None, **kwargs):
    """
    Lowlevel interface for plot_kde_1d_highlevel (see there for the remaining
    parameters).

    Parameters
    ----------
    df: pandas.DataFrame
        The rows are the observations, the columns the variables.
    w: pandas.DataFrame
        The corresponding weights.

    Returns
    -------
    ax: matplotlib axis
        Axis of the plot.
    """
    x_vals, pdf = kde_1d(df, w, x, xmin=xmin, xmax=xmax,  numx=numx, kde=kde)
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(x_vals, pdf, **kwargs)
    # TODO This fixes the upper bound inadequately
    # ax.set_ylim(bottom=min(ax.get_ylim()[0], 0))
    ax.set_xlabel(x)
    ax.set_ylabel("Posterior")
    ax.set_xlim(xmin, xmax)
    if title is not None:
        ax.set_title(title)
    if refval is not None:
        ax.axvline(refval[x], color=refval_color, linestyle='dotted')

    # set size
    if size is not None:
        ax.get_figure().set_size_inches(size)

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
        Defaults to 50.
    numy int, optional
        The number of bins in y direction.
        Defaults to 50.
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


def plot_kde_2d_highlevel(
        history: History, x: str, y: str, m: int = 0, t: int = None,
        xmin: float = None, xmax: float = None, ymin: float = None,
        ymax: float = None, numx: int = 50, numy: int = 50, ax=None,
        size=None, colorbar=True, title: str = None, refval=None,
        refval_color='C1', kde=None, **kwargs):
    """
    Plot 2d kernel density estimate of parameter samples.

    Parameters
    ----------
    history: History
        History to extract data from.
    x: str
        The variable for the x-axis.
    y: str
        The variable for the y-axis.
    m: int, optional
        Id of the model to plot for.
    t: int, optional
        Time point to plot for. Defaults to last time point.
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
    ax: matplotlib.axes.Axes, optional
        The axis object to use.
    size: 2-Tuple of float
        Size of the plot in inches.
    colorbar: bool, optional
        Whether to plot a colorbar. Defaults to True.
    title: str, optional
        Title for the plot. Defaults to None.
    refval: dict, optional
        A reference parameter to be shown in the plots. Default: None.
    refval_color: str, optional
        Color to use for the reference value.
    kde: pyabc.Transition, optional
        The kernel density estimator to use for creating a smooth density
        from the sample. If None, a multivariate normal kde with
        cross-validated scaling is used..

    Returns
    -------
    ax: matplotlib axis
        Axis of the plot.
    """
    df, w = history.get_distribution(m=m, t=t)

    return plot_kde_2d(
        df, w, x, y, xmin, xmax, ymin, ymax, numx, numy, ax, size, colorbar,
        title, refval, refval_color, kde, **kwargs)


def plot_kde_2d(df, w, x, y, xmin=None, xmax=None, ymin=None, ymax=None,
                numx=50, numy=50, ax=None, size=None, colorbar=True,
                title: str = None, refval=None, refval_color='C1', kde=None,
                **kwargs):
    """
    Plot a 2d kernel density estimate of parameter samples.

    Parameters
    ----------
    df: Pandas Dataframe
        The rows are the observations, the columns the variables
    w: The corresponding weights.

    For the other parameters, see `plot_kde_2d_highlevel`.

    Returns
    -------
    ax: matplotlib axis
        Axis of the plot.

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
        ax.scatter([refval[x]], [refval[y]], color=refval_color)

    # set size
    if size is not None:
        ax.get_figure().set_size_inches(size)

    return ax


def plot_kde_matrix_highlevel(
        history, m: int = 0, t: int = None, limits=None,
        colorbar: bool = True, height: float = 2.5,
        numx: int = 50, numy: int = 50, refval=None, refval_color='C1',
        kde=None, arr_ax=None):
    """
    Plot a KDE matrix for 1- and 2-dim marginals of the parameter samples.

    Parameters
    ----------
    history: History
        History to extract data from.
    m: int, optional
        Id of the model to plot for.
    t: int, optional
        Time point to plot for. Defaults to last time point.
    limits: dictionary, optional
        Dictionary of the form ``{"name": (lower_limit, upper_limit)}``.
    colorbar: bool
        Whether to plot the colorbars or not.
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
    refval_color: str, optional
        Color to use for the reference value.
    kde: pyabc.Transition, optional
        The kernel density estimator to use for creating a smooth density
        from the sample. If None, a multivariate normal kde with
        cross-validated scaling is used.
    arr_ax:
        Array of axes objects to use.

    Returns
    -------
    arr_ax:
        Array of the generated plots' axes.
    """
    df, w = history.get_distribution(m=m, t=t)

    return plot_kde_matrix(
        df, w, limits, colorbar, height, numx, numy, refval, refval_color,
        kde, arr_ax)


def plot_kde_matrix(df, w, limits=None, colorbar=True, height=2.5,
                    numx=50, numy=50, refval=None, refval_color='C1',
                    kde=None, arr_ax=None):
    """
    Plot a KDE matrix for 1- and 2-dim marginals of the parameter samples.

    Parameters
    ----------
    df: Pandas Dataframe
        The rows are the observations, the columns the variables.
    w: np.narray
        The corresponding weights.

    Other parameters: See plot_kde_matrix_highlevel.

    Returns
    -------
    arr_ax:
        Array of the generated plots' axes.
    """

    n_par = df.shape[1]
    par_names = list(df.columns.values)

    if arr_ax is None:
        fig, arr_ax = plt.subplots(nrows=n_par, ncols=n_par,
                                   sharex=False, sharey=False,
                                   figsize=(height * n_par, height * n_par))
    else:
        fig = arr_ax[0, 0].get_figure()

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
                    ax=ax, title=None, colorbar=colorbar,
                    refval=refval, refval_color=refval_color,
                    kde=kde)

    def scatter(x, y, ax):
        alpha = w / w.max()
        colors = np.zeros((alpha.size, 4))
        colors[:, 3] = alpha
        ax.scatter(x, y, color="k")
        if refval is not None:
            ax.scatter([refval[x.name]], [refval[y.name]], color=refval_color)
        ax.set_xlim(*limits.get(x.name, default))
        ax.set_ylim(*limits.get(y.name, default))

    def hist_1d(x, ax):
        df = pd.concat((x,), axis=1)
        plot_kde_1d(df, w, x.name,
                    xmin=limits.get(x.name, default)[0],
                    xmax=limits.get(x.name, default)[1],
                    numx=numx,
                    ax=ax, refval=refval, refval_color=refval_color,
                    kde=kde)

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
    format_plot_matrix(arr_ax, par_names)

    # adjust subplots to fit
    fig.tight_layout()

    return arr_ax
