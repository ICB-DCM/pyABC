import matplotlib.pyplot as plt

from ..storage import History


def plot_sample_numbers(histories, ax=None):
    # preprocess input
    if isinstance(histories, History):
        histories = [histories]
    if ax is None:
        _, ax = plt.subplots()
    n = len(histories)

    # extract sample numbers
    samples = []
    for history in histories:
        samples = np.array(history.get_all_populations()['samples'])


    return ax
