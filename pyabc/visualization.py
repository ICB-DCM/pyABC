"""
Visualizations
--------------

Helper functions to visualize results of ABCSMC runs.
"""
import numpy as np
from .transition import MultivariateNormalTransition, silverman_rule_of_thumb
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def kde_1d(df, w, x, xmin=None, xmax=None, numx=50):
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
        Defaults tp 50.

    Returns
    -------

    x, pdf: (np.ndarray, np.ndarray)
        The x and the densities at these points.
        These can be passed for plotting, for example as
        plt.plot(x, pdf)

    """
    kde = MultivariateNormalTransition(
        scaling=1,
        bandwidth_selector=silverman_rule_of_thumb)
    kde.fit(df[[x]], w)
    if xmin is None:
        xmin = df[x].min()
    if xmax is None:
        xmax = df[x].max()
    x_vals = np.linspace(xmin, xmax, num=numx)
    test = pd.DataFrame({x: x_vals})
    pdf = kde.pdf(test)
    return x_vals, pdf


def plot_kde_1d(df, w, x, xmin=None, xmax=None, numx=50, ax=None, **kwargs):
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

    Returns
    -------

    ax: matplotlib axis
        axis of the plot

    """
    x_vals, pdf = kde_1d(df, w, x, xmin=xmin, xmax=xmax,  numx=numx)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(x_vals, pdf, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel("Posterior")
    return ax


def kde_2d(df, w, x, y, xmin=None, xmax=None, ymin=None, ymax=None,
           numx=50, numy=50):
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

    Returns
    -------

    X, Y, PDF: (np.ndarray, np.ndarray, np.ndarray)
        The X, the Y and the densities at these points.
        These can be passed for plotting, for example as
        plt.pcolormesh(X, Y, PDF)

    """
    kde = MultivariateNormalTransition(
        scaling=1,
        bandwidth_selector=silverman_rule_of_thumb)
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
                title=True, **kwargs):
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
        Defaults tp 50.
    numy int, optional
        The number of bins in y direction.
        Defaults tp 50.
    colorbar: bool, optional
        Whether to plot a colorbar. Defaults to True.
    title: bool, optional
        Whether to put a title on the plot. Defaults to True.

    Returns
    -------

    ax: matplotlib axis
        axis of the plot

    """
    X, Y, PDF = kde_2d(df, w, x, y,
                       xmin=xmin, xmax=xmax,
                       ymin=ymin, ymax=ymax, numx=numx, numy=numy)
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    mesh = ax.pcolormesh(X, Y, PDF, **kwargs)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title("Posterior")
    if colorbar:
        cbar = fig.colorbar(mesh)
        cbar.set_label("PDF")
    return ax


def plot_kde_matrix(df, w, limits=None, colorbar=True):
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
    """
    grid = sns.PairGrid(df, diag_sharey=False)
    if limits is None:
        limits = {}

    default = (None, None)

    def off_diagonal(x, y, **kwargs):
        df = pd.concat((x, y), axis=1)
        plot_kde_2d(df, w,
                    x.name, y.name,
                    xmin=limits.get(x.name, default)[0],
                    xmax=limits.get(x.name, default)[1],
                    ymin=limits.get(y.name, default)[0],
                    ymax=limits.get(y.name, default)[1],
                    ax=plt.gca(), title=False, colorbar=colorbar)

    def scatter(x, y, **kwargs):
        alpha = w / w.max()
        colors = np.zeros((alpha.size, 4))
        colors[:, 3] = alpha
        plt.gca().scatter(x, y, color="k")
        plt.gca().set_xlim(*limits.get(x.name, default))
        plt.gca().set_ylim(*limits.get(y.name, default))

    def diagonal(x, **kwargs):
        df = pd.concat((x,), axis=1)
        plot_kde_1d(df, w, x.name,
                    xmin=limits.get(x.name, default)[0],
                    xmax=limits.get(x.name, default)[1],
                    ax=plt.gca())

    grid.map_diag(diagonal)
    grid.map_upper(scatter)
    grid.map_lower(off_diagonal)
    return grid
