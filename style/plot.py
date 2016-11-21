import matplotlib as mpl
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
import contextlib
import os
from .plotstyle import linewidth
from .labelpad import reduced

mpl.use("agg", force=True)


def strip_characters_for_save(string):
    return (string.replace("$", "").replace("^","").replace("{","").replace("}","").replace("/", "")
           .replace("(","") .replace(")","").replace(">",""))


vertical_label_angle = -90



annotarrowprops = dict(arrowstyle="-|>", facecolor="black")


def reduced_ticks_0_1(ax, axis="both", despine=True, label_pad_reduce=reduced):
    major_ticks = [0, 1]
    minor_ticks = [.2, .4, .6, .8]
    major_labels = [0, 1]

    if despine:
        sns.despine(ax=ax)

    if axis == "x" or axis == "both":
        ax.set_xticks(major_ticks)
        ax.set_xticklabels(major_labels)
        ax.set_xticks(minor_ticks, minor=True)
        ax.xaxis.labelpad = label_pad_reduce
        ax.set_xlim(0, 1)

    if axis == "y" or axis == "both":
        ax.set_yticks(major_ticks)
        ax.set_yticklabels(major_labels)
        ax.set_yticks(minor_ticks, minor=True)
        ax.yaxis.labelpad = label_pad_reduce
        ax.set_ylim(0, 1)


def rotate_pair_grid(g, rotation=45):
    for ax in g.axes[-1]:
        plt.setp( ax.xaxis.get_majorticklabels(), rotation=rotation)


def style_cbar_0_1(cbar, label=None, labelpad=reduced):
    if label is not None:
        cbar.set_label(label, labelpad=labelpad)
    cbar.set_ticks([0, .2, .4, .6, .8, 1])
    cbar.set_ticklabels([0, "", "", "", "", 1])
    cbar.outline.set_linewidth(0)


def make_cbar(ax, color_list, tick_labels, label=""):
    bounds = sp.arange(len(color_list)+1)

    cmap = mpl.colors.ListedColormap(color_list)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    cbar = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, boundaries=bounds, ticks=bounds + .5)
    cbar.set_ticklabels(tick_labels)
    cbar.outline.set_linewidth(0)
    cbar.set_label(label)
    return cbar


def vertical_xlabel(ax):
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=vertical_label_angle)


def colorbar(ax, mappable, label, nticks=4, labelpad=None):
    ticker = mpl.ticker.MaxNLocator(nticks)
    cb = ax.figure.colorbar(mappable, ticks=ticker)
    cb.outline.set_linewidth(0)
    cb.set_label(label, labelpad=labelpad)
    return cb


def annotate_group(ax, xmin, xmax, y, text, length=.1, text_offset=0, linewidth=linewidth):
    ax.plot([xmin, xmin, xmax, xmax],[y-length, y, y, y-length], color="k", clip_on=False, linewidth=linewidth, solid_capstyle="butt")
    ax.text((xmin+xmax)/2, y+text_offset, text,  va="bottom" if length >= 0 else "top", ha="center")


def save_convert_list(lst):
    if (lst.astype(int) == lst).all():
        return lst.astype(int)
    return lst


def middle_ticks_minor(ax, axis="both"):
    if axis == "x" or axis == "both":
        xticks = save_convert_list(ax.get_xticks())
        ax.set_xticks([xticks[0], xticks[-1]])
        ax.set_xticks(xticks[1:-1], minor=True)
        ax.set_xticklabels([xticks[0], xticks[-1]])

    if axis == "y" or axis == "both":
        yticks = save_convert_list(ax.get_yticks())
        ax.set_yticks([yticks[0], yticks[-1]])
        ax.set_yticks(yticks[1:-1], minor=True)
        ax.set_yticklabels(ax.get_yticks())


def remove_middle_tick_labels(ax, axis="both"):
    if axis == "x" or axis == "both":
        xticks = ax.get_xticklabels()
        ax.set_xticklabels(xticks[:1] + [""] * (len(xticks)-2) + xticks[-1:])

    if axis == "y" or axis == "both":
        yticks = ax.get_yticks()
        ax.set_yticklabels([str(yticks[0])] + [""] * (len(yticks) - 2) + [str(yticks[-1])])


def set_ticksmiddle_minor(ax, ticks, axis):
    ticks = list(ticks)
    if axis == "x":
        set_lim = ax.set_xlim
        set_ticks = ax.set_xticks
    if axis == "y":
        set_lim = ax.set_ylim
        set_ticks = ax.set_yticks

    set_lim(ticks[0], ticks[-1])
    set_ticks([ticks[0], ticks[-1]])
    set_ticks(ticks[1:-1], minor=True)


def despine(ax):
    sns.despine(ax=ax)


def no_despine(ax):
    pass


@contextlib.contextmanager
def plot(output_file: str, *args, despine=despine, **kwargs):
    fig, ax = plt.subplots(*args, **kwargs)
    yield (fig, ax)
    try:
        os.makedirs(os.path.dirname(output_file))
    except FileExistsError:
        pass
    despine(ax)

    fig.savefig(output_file, bbox_inches="tight", transparent=True)
    #plt.close(fig)
