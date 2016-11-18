import matplotlib as mpl
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt

from .plotstyle import linewidth
from .name import latex_name

mpl.use("agg", force=True)


def strip_characters_for_save(string):
    return (string.replace("$", "").replace("^","").replace("{","").replace("}","").replace("/", "")
           .replace("(","") .replace(")","").replace(">",""))


vertical_label_angle = -90


default_label_pad_reduced = -5

annotarrowprops = dict(arrowstyle="-|>", facecolor="black")


def reduced_ticks_0_1(ax, axis="both", despine=True, label_pad_reduce=default_label_pad_reduced):
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


def style_cbar_0_1(cbar, label=None, labelpad=default_label_pad_reduced):
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


def two_d_measure_scatter(df, ax, fig, x, y, hue, style_cbar=style_cbar_0_1,
                          vmin=0, vmax=1, s=20, tight_ax_lims=True, labelpad=None, clip_on=True,
                          with_cbar=True):

    mappable = ax.scatter(df[x], df[y], c=df[hue], s=s, edgecolor="none", vmin=vmin, vmax=vmax,clip_on=clip_on)

    if tight_ax_lims:
        ax.set_ylim(df[y].min(), df[y].max())
        ax.set_xlim(df[x].min(), df[x].max())
    if labelpad is not None:
        ax.set_xlabel(latex_name[x], labelpad=labelpad)
        ax.set_ylabel(latex_name[y], labelpad=labelpad)
    else:
        ax.set_xlabel(latex_name[x])
        ax.set_ylabel(latex_name[y])

    ax.locator_params(nbins=4)

    if with_cbar:
        cbar = fig.colorbar(mappable, ticks=mpl.ticker.MaxNLocator(4))
        style_cbar(cbar, latex_name[hue])
        cbar.outline.set_linewidth(0)
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


def ticks_to_major_minor(ax, axis="both"):
    if axis == "x" or axis == "both":
        xticks = ax.get_xticks()
        ax.set_xticks([xticks[0], xticks[-1]])
        ax.set_xticks(xticks[1:-1], minor=True)
        ax.set_xticklabels([xticks[0], xticks[-1]])

    if axis == "y" or axis == "both":
        yticks = ax.get_yticks()
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