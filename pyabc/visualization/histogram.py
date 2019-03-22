import matplotlib.pyplot as plt
import numpy as np


def plot_histogram_1d(
        history, x, m=0, t=None, xmin=None, xmax=None, **kwargs):
    fig, ax = plt.subplots()

    df, w = history.get_distribution(t=t)

    if xmin is not None and xmax is not None:
        range_ = (xmin, xmax)
    else:
        range_ = None

    # plot
    ax.hist(x=df[x], range=range_, weights=w, **kwargs)

    return ax


def plot_histogram_2d(
        history, x, y, m=0, t=None, xmin=None, xmax=None, ymin=None, ymax=None, **kwargs):
    fig, ax = plt.subplots()

    df, w = history.get_distribution(t=t)
    
    xrange_ = yrange_ = None
    if xmin is not None and xmax is not None:
        xrange_ = [xmin, xmax]
    if ymin is not None and ymax is not None:
        yrange_ = [ymin, ymax]
    if xrange_ and yrange_:
        range_ = [xrange_, yrange_]
    else:
        range_ = None

    ax.hist2d(x=df[x], y=df[y], range=range_, weights=w, **kwargs)

    return ax
