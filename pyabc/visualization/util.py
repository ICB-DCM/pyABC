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
