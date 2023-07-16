"""Model probability plots"""

from typing import TYPE_CHECKING

import matplotlib as mpl
import matplotlib.pyplot as plt

from ..storage import History

if TYPE_CHECKING:
    import plotly.graph_objs as go


def plot_model_probabilities(
    history: History,
    rotation: int = 0,
    title: str = "Model probabilities",
    size: tuple = None,
    ax: mpl.axes.Axes = None,
) -> mpl.axes.Axes:
    """
    Plot the probabilities of models over time.

    Parameters
    ----------

    history: History
        The history to extract data from.
    rotation:
        Rotation of x axis labels.
    title:
        Title of the plot.
    size:
        Size of the figure.
    ax:
        Matplotlib axes to plot on.

    Returns
    -------
    ax:
        The matplotlib axes on which the plot was created.
    """
    # create figure
    if ax is None:
        _, ax = plt.subplots()

    # extract model probabilities
    model_probabilities = history.get_model_probabilities()

    # displayed in plot legend
    model_probabilities.columns.name = "Model"

    # plot
    ax = model_probabilities.plot.bar(rot=rotation, legend=True, ax=ax)

    # format plot
    ax.set_ylabel("Probability")
    ax.set_xlabel("Population index")
    ax.set_title(title)

    if size is not None:
        ax.get_figure().set_size_inches(size)

    return ax


def plot_model_probabilities_plotly(
    history: History,
    rotation: int = 0,
    title: str = "Model probabilities",
    size: tuple = None,
    fig: "go.Figure" = None,
) -> "go.Figure":
    """Plot model probabilities using plotly."""
    import plotly.graph_objects as go

    # extract model probabilities
    model_probabilities = history.get_model_probabilities()

    # displayed in plot legend
    model_probabilities.columns.name = "Model"

    # plot
    if fig is None:
        fig = go.Figure()
    for model in model_probabilities.columns:
        fig.add_trace(
            go.Bar(
                x=model_probabilities.index,
                y=model_probabilities[model],
                name=model,
            )
        )

    # format plot
    fig.update_layout(
        title=title,
        xaxis_title="Population index",
        yaxis_title="Probability",
        xaxis_tickangle=rotation,
    )

    if size is not None:
        fig.update_layout(width=size[0], height=size[1])

    return fig
