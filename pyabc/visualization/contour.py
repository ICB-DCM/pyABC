"""Contour plots.

To plot contours from the weighted importance samples, the visualization
routines employ a kernel density estimate. Note that this can "over-smoothen"
so that local structure is lost. If this could be the case, it makes sense
to in the argument `kde` reduce the `scaling` in the default
MultivariateNormalTransition(), or to replace it by a GridSearchCV() to
automatically find a visually good level of smoothness.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..storage import History
from .kde import kde_2d, plot_kde_1d
from .util import format_plot_matrix


def plot_contour_2d(
    history: History,
    x: str,
    y: str,
    m: int = 0,
    t: int = None,
    xmin: float = None,
    xmax: float = None,
    ymin: float = None,
    ymax: float = None,
    numx: int = 50,
    numy: int = 50,
    ax=None,
    size=None,
    title: str = None,
    refval=None,
    refval_color='C1',
    kde=None,
    xname: str = None,
    yname: str = None,
    show_clabel: bool = False,
    show_legend: bool = False,
    clabel_kwargs: dict = None,
    **kwargs,
):
    """
    Plot 2d contour of parameter samples.

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
    numy int, optional
        The number of bins in y direction.
    ax: matplotlib.axes.Axes, optional
        The axis object to use.
    size: 2-Tuple of float
        Size of the plot in inches.
    title: str, optional
        Title for the plot. Defaults to None.
    refval: dict, optional
        A reference parameter to be shown in the plots. Default: None.
    refval_color: str, optional
        Color to use for the reference value.
    kde: pyabc.Transition, optional
        The kernel density estimator to use for creating a smooth density
        from the sample. If None, a multivariate normal kde with
        cross-validated scaling is used.
    xname:
        Parameter name for the x-axis.
    xname:
        Parameter name for the y-axis.
    show_clabel:
        Whether to show in-line contour labels.
    show_legend:
        Whether to show line labels in form of a legend.
    clabel_kwargs:
        Arguments to pass to `matplotlib.contour.ContourLabeler.clabel`.
        E.g.: "inline", "inline_spacing".

    Additional keyword arguments are passed to `matplotlib.axes.Axes.contour`.

    Returns
    -------
    ax: matplotlib axis
        Axis of the plot.
    """
    df, w = history.get_distribution(m=m, t=t)

    return plot_contour_2d_lowlevel(
        df=df,
        w=w,
        x=x,
        y=y,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        numx=numx,
        numy=numy,
        ax=ax,
        size=size,
        title=title,
        refval=refval,
        refval_color=refval_color,
        kde=kde,
        xname=xname,
        yname=yname,
        show_clabel=show_clabel,
        show_legend=show_legend,
        clabel_kwargs=clabel_kwargs,
        **kwargs,
    )


def plot_contour_2d_lowlevel(
    df,
    w,
    x,
    y,
    xmin=None,
    xmax=None,
    ymin=None,
    ymax=None,
    numx=50,
    numy=50,
    ax=None,
    size=None,
    title: str = None,
    refval=None,
    refval_color='C1',
    kde=None,
    xname: str = None,
    yname: str = None,
    show_clabel: bool = False,
    show_legend: bool = False,
    clabel_kwargs: dict = None,
    **kwargs,
):
    """
    Plot a 2d contour of parameter samples.

    Parameters
    ----------
    df: Pandas Dataframe
        The rows are the observations, the columns the variables
    w: The corresponding weights.

    For the other parameters, see `plot_contour_2d`.

    Returns
    -------
    ax: matplotlib axis
        Axis of the plot.

    """
    X, Y, PDF = kde_2d(
        df=df,
        w=w,
        x=x,
        y=y,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        numx=numx,
        numy=numy,
        kde=kde,
    )
    if xname is None:
        xname = x
    if yname is None:
        yname = y
    if ax is None:
        _, ax = plt.subplots()

    # plot contours
    contour = ax.contour(X, Y, PDF, **kwargs)

    # show in-line contour labels
    if show_clabel:
        if clabel_kwargs is None:
            clabel_kwargs = {}
        ax.clabel(contour, **clabel_kwargs)

    # show legend
    if show_legend:
        handles, labels = contour.legend_elements("pdf")
        ax.legend(handles, labels)

    # show title
    if title is not None:
        ax.set_title(title)

    # show reference value
    if refval is not None:
        ax.scatter([refval[x]], [refval[y]], color=refval_color)

    # label axes
    ax.set_xlabel(xname)
    ax.set_ylabel(yname)

    # set size
    if size is not None:
        ax.get_figure().set_size_inches(size)

    return ax


def plot_contour_matrix(
    history,
    m: int = 0,
    t: int = None,
    limits: dict = None,
    height: float = 2.5,
    numx: int = 50,
    numy: int = 50,
    refval: dict = None,
    refval_color='C1',
    kde=None,
    names: dict = None,
    show_clabel: bool = False,
    show_legend: bool = False,
    clabel_kwargs: dict = None,
    arr_ax=None,
    **kwargs,
):
    """
    Plot a contour matrix for 1- and 2-dim marginals of the parameter samples.

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
    names:
        Parameter names to use.
    show_clabel:
        Whether to show in-line contour labels.
    show_legend:
        Whether to show line labels in form of a legend.
    clabel_kwargs:
        Arguments to pass to `matplotlib.contour.ContourLabeler.clabel`.
        E.g.: "inline", "inline_spacing".
    arr_ax:
        Array of axes objects to use.

    Returns
    -------
    arr_ax:
        Array of the generated plots' axes.
    """
    df, w = history.get_distribution(m=m, t=t)

    return plot_contour_matrix_lowlevel(
        df=df,
        w=w,
        limits=limits,
        height=height,
        numx=numx,
        numy=numy,
        refval=refval,
        refval_color=refval_color,
        kde=kde,
        names=names,
        show_clabel=show_clabel,
        show_legend=show_legend,
        clabel_kwargs=clabel_kwargs,
        arr_ax=arr_ax,
        **kwargs,
    )


def plot_contour_matrix_lowlevel(
    df: pd.DataFrame,
    w: np.ndarray,
    limits: dict = None,
    height: int = 2.5,
    numx: int = 50,
    numy: int = 50,
    refval: dict = None,
    refval_color='C1',
    kde=None,
    names: dict = None,
    show_clabel: bool = False,
    show_legend: bool = False,
    clabel_kwargs: dict = None,
    arr_ax=None,
    **kwargs,
):
    """
    Plot a contour matrix for 1- and 2-dim marginals of the parameter samples.

    Parameters
    ----------
    df: Pandas Dataframe
        The rows are the observations, the columns the variables.
    w: np.narray
        The corresponding weights.

    Other parameters: See plot_contour_matrix.

    Returns
    -------
    arr_ax:
        Array of the generated plots' axes.
    """

    n_par = df.shape[1]
    par_ids = list(df.columns.values)

    if names is None:
        names = {key: key for key in par_ids}
    if arr_ax is None:
        fig, arr_ax = plt.subplots(
            nrows=n_par,
            ncols=n_par,
            sharex=False,
            sharey=False,
            figsize=(height * n_par, height * n_par),
        )
    else:
        fig = arr_ax[0, 0].get_figure()

    if limits is None:
        limits = {}
    default = (None, None)

    def hist_2d(x, y, ax):
        df = pd.concat((x, y), axis=1)
        plot_contour_2d_lowlevel(
            df,
            w,
            x.name,
            y.name,
            xmin=limits.get(x.name, default)[0],
            xmax=limits.get(x.name, default)[1],
            ymin=limits.get(y.name, default)[0],
            ymax=limits.get(y.name, default)[1],
            numx=numx,
            numy=numy,
            ax=ax,
            title=None,
            refval=refval,
            refval_color=refval_color,
            kde=kde,
            xname=names[x.name],
            yname=names[y.name],
            show_clabel=show_clabel,
            show_legend=show_legend,
            clabel_kwargs=clabel_kwargs,
            **kwargs,
        )

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
        plot_kde_1d(
            df,
            w,
            x.name,
            xmin=limits.get(x.name, default)[0],
            xmax=limits.get(x.name, default)[1],
            numx=numx,
            ax=ax,
            refval=refval,
            refval_color=refval_color,
            kde=kde,
            xname=x.name,
        )

    # fill all subplots
    for i in range(0, n_par):
        y_name = par_ids[i]
        y = df[y_name]

        # diagonal
        ax = arr_ax[i, i]
        hist_1d(y, ax)

        for j in range(0, i):
            x_name = par_ids[j]
            x = df[x_name]

            # lower
            ax = arr_ax[i, j]
            hist_2d(x, y, ax)

            # upper
            ax = arr_ax[j, i]
            scatter(y, x, ax)

    # format
    format_plot_matrix(arr_ax, [names[key] for key in par_ids])

    # adjust subplots to fit
    fig.tight_layout()

    return arr_ax
