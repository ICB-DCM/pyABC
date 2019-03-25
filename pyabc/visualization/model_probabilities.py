from ..storage import History


def plot_model_probabilities(
        history: History,
        rotation: int = 0,
        title: str = "Model probabilities"):
    """
    Plot the probabilities of models over time.

    Parameters
    ----------

    history: History
        The history to extract data from.
    rotation: int, optional (default = 0)
        Rotation of x axis labels.
    title: str, optional
        Title of the plot.
    """
    # extract model probabilities
    model_probabilities = history.get_model_probabilities()

    # displayed in plot legend
    model_probabilities.columns.name = "Model"

    # plot
    ax = model_probabilities.plot.bar(rot=rotation, legend=True)

    # format plot
    ax.set_ylabel("Probability")
    ax.set_xlabel("Population index")
    ax.set_title(title)

    return ax
