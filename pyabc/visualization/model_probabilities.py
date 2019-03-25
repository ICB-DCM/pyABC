import matplotlib.pyplot as plt

from ..storage import History


def plot_model_probabilities(
        history: History,
        rotation: int = 0):
    model_probabilities = history.get_model_probabilities()

    # displayed in plot legend
    model_probabilities.columns.name = "Model"
    
    ax = model_probabilities.plot.bar(rot=rotation, legend=True)

    ax.set_ylabel("Probability")
    ax.set_xlabel("Population index")
    ax.set_title("Model probabilities")

    return ax
