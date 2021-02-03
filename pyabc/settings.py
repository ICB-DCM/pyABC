"""Global settings"""

import matplotlib as mpl
from matplotlib import rcParams
from typing import List


def set_figure_params(
    style: str = 'pyabc',
    color_map: str = 'plasma',
    color_cycle: List[str] = None,
):
    """Set global figure parameters for a consistent, beautified design.

    Parameters
    ----------
    style:
        Style set to use. Possible: 'pyabc', 'default'
        (the matplotlib defaults).
    color_map:
        Colormap ('image.cmap').
    color_cycle:
        Color cycle ('axes.prop_cycle').

    **Note:** This modifies global matplotlib parameters.
    """
    style = style.lower()
    if style == 'pyabc':
        _set_figure_params_pyabc(color_map=color_map, color_cycle=color_cycle)
    elif style == 'default':
        _set_figure_params_default()
    else:
        raise ValueError(f"Style not recognized: {style}")


def _set_figure_params_pyabc(color_map: str, color_cycle: List[str]):
    """Set layout parameters for style 'pyabc'"""
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    if color_map is not None:
        rcParams['image.cmap'] = color_map
    if color_cycle is not None:
        rcParams['axes.prop_cycle'] = color_cycle


def _set_figure_params_default():
    """Set layout parameters for the default matplotlib style"""
    mpl.rcdefaults()
