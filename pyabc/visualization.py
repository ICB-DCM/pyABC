"""
Visualizations
--------------

Helper functions to visualize results of ABCSMC runs.
"""
import numpy as np
import pandas as pd
from .transition import MultivariateNormalTransition, silverman_rule_of_thumb
import matplotlib.pyplot as plt


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
                numx=50, numy=50, ax=None):
    """
    Plots a 2d histogram.

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

    ax: matplotlib axis
        axis of the plot

    """
    X, Y, PDF = kde_2d(df, w, x, y,
                       xmin=xmin, xmax=xmax,
                       ymin=ymin, ymax=ymax, numx=numx, numy=numy)
    if ax is None:
        fig, ax = plt.subplots()
    mesh = ax.pcolormesh(X, Y, PDF)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title("Posterior")
    cbar = fig.colorbar(mesh)
    cbar.set_label("PDF")
    return ax
