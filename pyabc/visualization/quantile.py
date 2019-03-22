import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Union, List
import numpy as np

from ..storage import History
from .util import to_lists_or_default


def plot_quantiles(
        history: History,
        par_names: List = None,
        quantiles: List = None,
        size: tuple = None,):
    
    if par_names is None:
        df, _ = history.get_distribution()
        par_names = list(df.columns.values)
    if quantiles is None:
        quantiles = [0.95]

    # iterate over time points
    fig, ax = plt.subplots()
