import matplotlib as mpl
import seaborn as sns
from .size import m as m_size

linewidth = 1
bar_line_width = 3
font_size = 10
major_tick_size = 3
minor_tick_size = major_tick_size / 2
major_tick_pad = 1
axes_line_width = .75
mpl_style_pars = {
    'axes.labelsize': font_size,
    'axes.labelpad': 1,
    'axes.titlesize': font_size,
    'axes.linewidth': axes_line_width,
    'font.size': font_size,
    'xtick.labelsize': font_size,
    'ytick.labelsize': font_size,
    'legend.fontsize': font_size,
    'figure.figsize': m_size,
    'lines.linewidth': linewidth,
    'image.cmap': "viridis",
    'font.family': "Liberation Sans",
    'font.sans-serif': "arial",
    'xtick.major.size': major_tick_size,
    'ytick.major.size': major_tick_size,
    'xtick.minor.size': minor_tick_size,
    'ytick.minor.size': minor_tick_size,
    'xtick.major.width': axes_line_width,
    'xtick.minor.width': axes_line_width/2,
    'ytick.major.width': axes_line_width,
    'ytick.minor.width': axes_line_width/2,
    'xtick.major.pad': major_tick_pad,
    'ytick.major.pad': major_tick_pad,
    "legend.labelspacing": .2,
    "legend.handlelength": 1,
    "legend.handletextpad": .2,
    "legend.borderpad": 0,
    "legend.borderaxespad": .5,
    "axes.formatter.useoffset": False,
    "lines.color": "black"
}


sns.set_style('ticks')

mpl.rcParams.update(mpl_style_pars)
