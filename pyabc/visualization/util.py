from ..storage import History


def to_lists_or_default(histories, labels=None):
    """
    Interpret input using default values, and convert to lists.
    """
    if isinstance(histories, History):
        histories = [histories]

    if isinstance(labels, str):
        labels = [labels]
    elif labels is None:
        labels = ["History " + str(j) for j in range(n_history)]

    return histories, labels
