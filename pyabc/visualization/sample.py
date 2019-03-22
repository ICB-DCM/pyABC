import matplotlib.pyplot as plt

from ..storage import History


def plot_sample_numbers(histories, ids=None, ax=None):
    if isinstance(histories, History):
        histories = [histories]
    n_history = len(histories)
    if ids is None:
        ids = [1] * n_history
    if ax is None:
        _, ax = plt.subplots()
    
    # extract sample numbers
    n = len(histories)


    return ax


def get_list_of_histories_and_ids(histories, ids=None):
    """
    Make sure both are lists of the same length.
    If ids is None,  
    """
    if isinstance(histories, History):
        histories = [histories]
    n_history = len(histories)
    if ids is None:
        ids = [None] * n_history
    n_id = len(ids)
    if n_history == 1 and n_id > 1:
        histories =  [histories[0]] * n_id
    if n_id == 1 and n_history > 1:
        ids = [ids[0]] * n_history

    return histories, ids
