from ..storage import History


def to_lists_or_default(histories, labels=None):
    """
    Interpret input using default values, and convert to lists of the same
    length.
    """
    if isinstance(histories, History):
        histories = [histories]

    if isinstance(labels, str):
        labels = [labels]
    elif labels is None:
        labels = ["History " + str(j) for j in range(len(histories))]

    if len(histories) != len(labels):
        raise ValueError("The lengths of histories and labels do not match.")

    return histories, labels


def format_plot_matrix(arr_ax, par_names):
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
