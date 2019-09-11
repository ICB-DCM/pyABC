import matplotlib.pyplot as plt
import pandas as pd

from ..storage import History


def plot_histogram_1d(
        history: History, x: str, m: int = 0, t: int = None,
        xmin=None, xmax=None, ax=None, size=None, refval=None, **kwargs):
    """
    Plot 1d histogram of parameter samples.

    Parameters
    ----------

    history: History
        History to extract data from.
    x: str
        Id of the parameter to plot for.
    m: int, optional (default = 0)
        Id of the model to plot for.
    t: int, optional (default = None, i.e. the last time)
        Time point to plot for.
    xmin, xmax: float
        Bounds for x. Both must be specified for bounds to be applied.
    ax: matplotlib.axes.Axes
        Axis object for the plot. If None is passed, a new figure is created.
    size: 2-Tuple of float, optional
        Size of the plot in inches.
    refval: dict, optional (default = None)
        A reference value for x, to be highlighted in the plot.

    Returns
    -------

    ax: Axis of the generated plot.
    """
    df, w = history.get_distribution(m=m, t=t)

    return plot_histogram_1d_lowlevel(
        df, w, x, xmin, xmax, ax=ax, size=size, refval=refval, **kwargs)


def plot_histogram_1d_lowlevel(
        df: pd.DataFrame, w: pd.DataFrame,
        x: str, xmin=None, xmax=None, ax=None, size=None, refval=None,
        **kwargs):
    """
    Lowlevel interface for plot_histogram_1d (see there for the remaining
    parameters).

    Parameters
    ----------

    df: pd.DataFrame
        Contains the parameters. Must have a column 'x'.
    w: pd.DataFrame
        Parameter weights.
    """

    if ax is None:
        _, ax = plt.subplots()

    if xmin is not None and xmax is not None:
        range_ = (xmin, xmax)
    else:
        range_ = None
    if refval is not None:
        ax.axvline(refval[x], color='C1', linestyle='dashed')

    # plot
    ax.hist(x=df[x], range=range_, weights=w, density=True, **kwargs)
    ax.set_xlabel(x)

    # set size
    if size is not None:
        ax.get_figure().set_size_inches(size)

    return ax


def plot_histogram_2d(
        history: History, x: str, y: str, m: int = 0, t: int = None,
        xmin=None, xmax=None, ymin=None, ymax=None, ax=None, size=None,
        refval=None, **kwargs):
    """
    Plot 2d histogram of parameter pair samples.

    Parameters
    ----------

    history: History
        History to extract data from.
    x, y: str
        Ids of the parameters to plot for.
    m: int, optional (default = 0)
        Id of the model to plot for.
    t: int, optional (default = None, i.e. the last time)
        Time point to plot for.
    xmin, xmax, ymin, ymax: float
        Bounds for x and y. All must be specified for bounds to be applied.
    ax: matplotlib.axes.Axes
        Axis object for the plot. If None is passed, a new figure is created.
    size: 2-Tuple of float, optional
        Size of the plot in inches.
    refval: dict, optional (default = None)
        Reference values for x and y, to be highlighted in the plot.

    Returns
    -------

    ax: Axis of the generated plot.
    """
    df, w = history.get_distribution(m=m, t=t)

    return plot_histogram_2d_lowlevel(
        df, w, x, y, xmin, xmax, ymin, ymax, ax=ax, size=size, refval=refval,
        **kwargs)


def plot_histogram_2d_lowlevel(
        df: pd.DataFrame, w: pd.DataFrame,
        x, y, xmin=None, xmax=None, ymin=None, ymax=None, ax=None,
        size=None, refval=None, **kwargs):
    """
    Lowlevel interface for plot_histogram_2d (see there for the remaining
    parameters).

    Parameters
    ----------

    df: pd.DataFrame
        Contains the parameters. Must have a column 'x'.
    w: pd.DataFrame
        Parameter weights.
    """
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
    ax.hist2d(x=df[x], y=df[y], range=range_, weights=w, density=True,
              **kwargs)
    if refval is not None:
        ax.scatter([refval[x]], [refval[y]], color='C1')
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    # set size
    if size is not None:
        ax.get_figure().set_size_inches(size)

    return ax


def plot_histogram_matrix(
        history: History, m: int = 0, t: int = None, size=None, refval=None,
        **kwargs):
    """
    Plot matrix of 1d and 2d histograms over all parameters.

    Parameters
    ----------

    history: History
        History to extract data from.
    m: int, optional (default = 0)
        Id of the model to plot for.
    t: int, optional (default = None, i.e. the last time)
        Time point to plot for.
    size: 2-Tuple of float, optional
        Size of the plot in inches.
    refval: dict, optional (default = None)
        Reference parameter values, to be highlighted in the plot.

    Returns
    -------

    arr_ax: list of matplotlib.axis.Axis
        Axis objects of the generated plots.
    """
    df, w = history.get_distribution(m=m, t=t)

    return plot_histogram_matrix_lowlevel(df, w, refval, **kwargs)


def plot_histogram_matrix_lowlevel(
        df: pd.DataFrame, w: pd.DataFrame, size=None, refval=None, **kwargs):
    """
    Lowlevel interface for plot_histogram_matrix (see there for the remaining
    parameters).

    Parameters
    ----------

    df: pd.DataFrame
        Contains the parameters. Must have a column 'x'.
    w: pd.DataFrame
        Parameter weights.
    """
    n_par = df.shape[1]
    par_names = list(df.columns.values)

    # create new figure
    fig, arr_ax = plt.subplots(
        nrows=n_par, ncols=n_par, sharex=False, sharey=False)

    def scatter(x, y, ax, refval=None):
        ax.scatter(x, y, color="k")
        if refval is not None:
            ax.scatter([refval[x.name]], [refval[y.name]], color='C1')

    # fill all subplots
    for i in range(0, n_par):
        y_name = par_names[i]
        y = df[y_name]

        # diagonal
        ax = arr_ax[i, i]
        plot_histogram_1d_lowlevel(df, w, y_name, ax=ax, refval=refval,
                                   **kwargs)

        for j in range(0, i):
            x_name = par_names[j]
            x = df[x_name]

            # lower
            ax = arr_ax[i, j]
            plot_histogram_2d_lowlevel(df, w, x_name, y_name, ax=ax,
                                       refval=refval, **kwargs)

            # upper
            ax = arr_ax[j, i]
            scatter(y, x, ax, refval=refval)

    # format
    _format_histogram_matrix(arr_ax, par_names)

    # set size
    if size is not None:
        ax.get_figure().set_size_inches(size)

    fig.tight_layout()

    return arr_ax


def _format_histogram_matrix(arr_ax, par_names):
    """
    Apply some post-formatting to tidy up the plot.
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
