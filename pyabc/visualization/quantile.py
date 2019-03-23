import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Union, List
import numpy as np

from ..storage import History
from .util import to_lists_or_default


def plot_quantiles(
        history: History,
        m: int = 0,
        par_names: List = None,
        quantiles: List = None,
        show_mean: bool = False,
        size: tuple = None,):

    if quantiles is None:
        quantiles = [0.5, 0.95]
    if par_names is None:
        df, _ = history.get_distribution(m=m)
        par_names = list(df.columns.values)

    n_par = len(par_names)
    n_pop = history.max_t + 1

    # prepare axes
    fig, arr_ax = plt.subplots(nrows=n_par, ncols=1, sharex=False, sharey=False)
    
    # prepare quantile matrix
    alphas = quantiles_to_alphas(quantiles)

    qs = np.empty((n_par, n_pop, n_alpha))

    for t in range(0, history.max_t + 1):
        df, w = history.get_distribution(m=m, t=t)
        # create par_names on the fly
        if par_names is None:
            par_names = list(df.colums.values)
        if n_par is 
        for i_par, par in enumerate(par_names):
            points = np.array(df[par])
            mean_t_par = np.sum(weights * points)
            mean[ipar, t] = mean_t_par
            for iquantile, quantile in enumerate(quantiles):
                cis[ipar, t, iquantile] = 

def quantiles_to_alphas(quantiles):
    alphas = []
    for quantile in quantiles:
        lower = 1.0 - quantile
